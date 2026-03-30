from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .dataset_adapters import get_dataset_adapter
from .model_runner import TransformersChatRunner
from .registry import ExperimentRegistry
from .reporting import RunReportWriter
from .specs import EvaluationRequest, PreparedSample
from .utils import batched, collect_environment_metadata, json_ready, set_reproducibility


def evaluate_experiment(
    request: EvaluationRequest,
    registry: ExperimentRegistry,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root or Path.cwd())
    _validate_request(request, registry)
    set_reproducibility(request.seed)

    environment_payload = collect_environment_metadata(project_root)
    registry_payload = {
        "models": {
            name: registry.models[name].to_dict() for name in request.model_names
        },
        "datasets": {
            name: registry.datasets[name].to_dict() for name in request.dataset_names
        },
    }
    report_writer = RunReportWriter(
        output_root=request.output_root,
        run_name=request.run_name,
        request_payload=request.to_dict(),
        environment_payload=environment_payload,
        registry_payload=registry_payload,
    )

    pair_summaries: list[dict[str, Any]] = []
    for model_name in request.model_names:
        model_spec = registry.models[model_name]
        runner = TransformersChatRunner(model_spec).load()
        try:
            for dataset_name in request.dataset_names:
                dataset_spec = registry.datasets[dataset_name]
                adapter = get_dataset_adapter(dataset_spec.adapter_name)
                dataset_limit = request.per_dataset_limit.get(
                    dataset_name,
                    request.default_dataset_limit,
                )
                prepared_samples, dataset_metadata = adapter.load(dataset_spec, limit=dataset_limit)
                _apply_system_prompt_override(
                    prepared_samples=prepared_samples,
                    dataset_name=dataset_name,
                    request=request,
                )
                sample_results = _evaluate_pair(
                    runner=runner,
                    adapter=adapter,
                    prepared_samples=prepared_samples,
                    request=request,
                    model_name=model_name,
                    dataset_name=dataset_name,
                )
                summary = adapter.aggregate(sample_results)
                pair_payload = {
                    "model": runner.describe(),
                    "dataset": dataset_spec.to_dict(),
                    "dataset_source": dataset_metadata,
                    "generation": request.generation.to_dict(),
                    "seed": request.seed,
                    "system_prompt_override": request.system_prompt_overrides.get(dataset_name),
                }
                report_writer.write_pair(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    pair_payload=pair_payload,
                    sample_results=sample_results,
                    summary=summary,
                )
                pair_summaries.append(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        **summary,
                    }
                )
        finally:
            runner.unload()

    final_report = report_writer.finalize()
    final_report["pair_summaries"] = pair_summaries
    return json_ready(final_report)


def _evaluate_pair(
    runner: TransformersChatRunner,
    adapter: Any,
    prepared_samples: list[PreparedSample],
    request: EvaluationRequest,
    model_name: str,
    dataset_name: str,
) -> list[dict[str, Any]]:
    sample_results: list[dict[str, Any]] = []
    description = f"{model_name} | {dataset_name}"

    for batch_samples in tqdm(
        batched(prepared_samples, request.generation.batch_size),
        desc=description,
        leave=False,
    ):
        batch_samples = list(batch_samples)
        outputs = runner.generate_batch(
            [sample.messages for sample in batch_samples],
            generation=request.generation,
        )
        for sample, output in zip(batch_samples, outputs):
            predicted_answer = adapter.extract_prediction(output["generated_text"], sample)
            metric = adapter.score_sample(
                sample=sample,
                predicted_answer=predicted_answer,
                generated_text=output["generated_text"],
            )
            sample_results.append(
                {
                    "sample_id": sample.sample_id,
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "messages": sample.messages,
                    "rendered_prompt": output["rendered_prompt"],
                    "reference_text": sample.reference_text,
                    "reference_answer": sample.reference_answer,
                    "predicted_answer": predicted_answer,
                    "generated_text": output["generated_text"],
                    "metric": metric,
                    "sample_metadata": sample.metadata,
                    "raw_sample": sample.raw_sample,
                }
            )

    return sample_results


def _validate_request(request: EvaluationRequest, registry: ExperimentRegistry) -> None:
    missing_models = [name for name in request.model_names if name not in registry.models]
    missing_datasets = [name for name in request.dataset_names if name not in registry.datasets]
    if missing_models:
        raise KeyError(f"Khong tim thay model trong registry: {missing_models}")
    if missing_datasets:
        raise KeyError(f"Khong tim thay dataset trong registry: {missing_datasets}")


def _apply_system_prompt_override(
    prepared_samples: list[PreparedSample],
    dataset_name: str,
    request: EvaluationRequest,
) -> None:
    override = request.system_prompt_overrides.get(dataset_name)
    if not override:
        return
    for sample in prepared_samples:
        if sample.messages and sample.messages[0].get("role") == "system":
            sample.messages[0]["content"] = override
