from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from .metrics import (
    compare_choice_answers,
    compare_numeric_or_text,
    compare_reasoning_answers,
    extract_final_answer_reasoning,
    extract_gsm8k_ground_truth,
    extract_gsm8k_prediction,
    extract_mathqa_choice,
    summarize_accuracy,
)
from .specs import DatasetSpec, PreparedSample
from .utils import find_first_existing_path, json_ready


ACE_REASON_SYSTEM_PROMPT = """You are a helpful assistant skilled in problem-solving. When given a task:
- Think through the problem step by step.
- Provide a clear, well-structured solution.
- End with the final answer clearly stated."""


MATH_SYSTEM_PROMPT = """You are a helpful assistant skilled in mathematical problem-solving. When given a math problem:
- Think through the problem step by step.
- Show your reasoning clearly.
- End with the final answer clearly stated."""


class BaseDatasetAdapter(ABC):
    name: str
    default_system_prompt: str

    def load(self, spec: DatasetSpec, limit: int | None = None) -> tuple[list[PreparedSample], dict[str, Any]]:
        raw_records, source_metadata = self.load_records(spec, limit=limit)
        prepared_samples = [
            self.prepare_sample(record=record, index=index)
            for index, record in enumerate(raw_records)
        ]
        source_metadata["prepared_sample_count"] = len(prepared_samples)
        source_metadata["adapter_name"] = self.name
        return prepared_samples, source_metadata

    @abstractmethod
    def load_records(self, spec: DatasetSpec, limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def prepare_sample(self, record: dict[str, Any], index: int) -> PreparedSample:
        raise NotImplementedError

    @abstractmethod
    def extract_prediction(self, generated_text: str, sample: PreparedSample) -> str:
        raise NotImplementedError

    @abstractmethod
    def score_sample(
        self,
        sample: PreparedSample,
        predicted_answer: str,
        generated_text: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def aggregate(self, sample_results: list[dict[str, Any]]) -> dict[str, Any]:
        return summarize_accuracy(sample_results)

    def _dataset_to_records(
        self,
        dataset: Dataset | list[dict[str, Any]],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(dataset, list):
            records = dataset[:limit] if limit is not None else dataset
            return records
        if limit is not None:
            limit = min(limit, len(dataset))
            dataset = dataset.select(range(limit))
        return [dict(item) for item in dataset]

    def _load_saved_dataset(
        self,
        path: Path,
        split: str,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        loaded = load_from_disk(str(path))
        if isinstance(loaded, DatasetDict):
            dataset = loaded[split]
            fingerprint = dataset._fingerprint
            available_splits = list(loaded.keys())
        else:
            dataset = loaded
            fingerprint = getattr(dataset, "_fingerprint", None)
            available_splits = []
        records = self._dataset_to_records(dataset, limit=limit)
        return records, {
            "source_kind": "local_disk",
            "resolved_path": str(path),
            "split": split,
            "available_splits": available_splits,
            "dataset_fingerprint": fingerprint,
            "num_loaded_records": len(records),
        }

    def _load_json_file(
        self,
        path: Path,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list JSON payload at {path}")
        records = payload[:limit] if limit is not None else payload
        return records, {
            "source_kind": "local_json",
            "resolved_path": str(path),
            "num_loaded_records": len(records),
        }

    def _load_hf_dataset(
        self,
        spec: DatasetSpec,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        dataset = load_dataset(
            spec.hf_dataset_id,
            spec.hf_subset,
            split=spec.split,
            revision=spec.revision,
        )
        records = self._dataset_to_records(dataset, limit=limit)
        return records, {
            "source_kind": "hf_dataset",
            "hf_dataset_id": spec.hf_dataset_id,
            "hf_subset": spec.hf_subset,
            "split": spec.split,
            "revision": spec.revision,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "num_loaded_records": len(records),
        }

    def _load_json_dataset_file(
        self,
        path: Path,
        split: str,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        dataset = load_dataset("json", data_files={split: str(path)}, split=split)
        records = self._dataset_to_records(dataset, limit=limit)
        return records, {
            "source_kind": "local_json_dataset",
            "resolved_path": str(path),
            "split": split,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "num_loaded_records": len(records),
        }


class AceReasonAdapter(BaseDatasetAdapter):
    name = "ace_reason"
    default_system_prompt = ACE_REASON_SYSTEM_PROMPT

    def load_records(self, spec: DatasetSpec, limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        local_path = find_first_existing_path(spec.local_paths)
        if local_path is not None:
            return self._load_saved_dataset(local_path, split=spec.split, limit=limit)
        if spec.hf_dataset_id:
            return self._load_hf_dataset(spec, limit=limit)
        raise FileNotFoundError(f"Khong tim thay dataset cho {spec.name}")

    def prepare_sample(self, record: dict[str, Any], index: int) -> PreparedSample:
        user_prompt = _extract_user_prompt(record)
        reference_text = _extract_assistant_text(record)
        reference_answer = extract_final_answer_reasoning(reference_text)
        sample_id = str(record.get("id") or record.get("uuid") or index)
        return PreparedSample(
            sample_id=sample_id,
            messages=[
                {"role": "system", "content": self.default_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            reference_text=reference_text,
            reference_answer=reference_answer,
            raw_sample=json_ready(record),
            metadata={"prompt_source": "messages_or_heuristic_fields"},
        )

    def extract_prediction(self, generated_text: str, sample: PreparedSample) -> str:
        return extract_final_answer_reasoning(generated_text)

    def score_sample(
        self,
        sample: PreparedSample,
        predicted_answer: str,
        generated_text: str,
    ) -> dict[str, Any]:
        is_correct, detail = compare_reasoning_answers(predicted_answer, sample.reference_answer)
        return {
            "metric_name": "accuracy",
            "metric_value": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "detail": detail,
        }


class GSM8KAdapter(BaseDatasetAdapter):
    name = "gsm8k"
    default_system_prompt = MATH_SYSTEM_PROMPT

    def load_records(self, spec: DatasetSpec, limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        local_path = find_first_existing_path(spec.local_paths)
        if local_path is not None:
            if local_path.is_dir():
                try:
                    return self._load_saved_dataset(local_path, split=spec.split, limit=limit)
                except Exception:
                    candidate_files = [
                        local_path / f"{spec.split}.json",
                        local_path / f"{spec.split}.jsonl",
                        local_path / "test.json",
                        local_path / "test.jsonl",
                    ]
                    for candidate in candidate_files:
                        if candidate.exists():
                            return self._load_json_dataset_file(candidate, split=spec.split, limit=limit)
            if local_path.suffix.lower() in {".json", ".jsonl"}:
                return self._load_json_dataset_file(local_path, split=spec.split, limit=limit)
        if spec.hf_dataset_id:
            return self._load_hf_dataset(spec, limit=limit)
        raise FileNotFoundError(f"Khong tim thay dataset cho {spec.name}")

    def prepare_sample(self, record: dict[str, Any], index: int) -> PreparedSample:
        question = _first_non_empty_value(record, ["question", "problem", "Problem", "prompt"])
        answer_text = _first_non_empty_value(record, ["answer", "solution", "output"])
        reference_answer = extract_gsm8k_ground_truth(answer_text)
        sample_id = str(record.get("id") or record.get("idx") or index)
        return PreparedSample(
            sample_id=sample_id,
            messages=[
                {"role": "system", "content": self.default_system_prompt},
                {"role": "user", "content": question},
            ],
            reference_text=answer_text,
            reference_answer=reference_answer,
            raw_sample=json_ready(record),
            metadata={},
        )

    def extract_prediction(self, generated_text: str, sample: PreparedSample) -> str:
        return extract_gsm8k_prediction(generated_text)

    def score_sample(
        self,
        sample: PreparedSample,
        predicted_answer: str,
        generated_text: str,
    ) -> dict[str, Any]:
        is_correct, detail = compare_numeric_or_text(predicted_answer, sample.reference_answer)
        return {
            "metric_name": "accuracy",
            "metric_value": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "detail": detail,
        }


class MathQAAdapter(BaseDatasetAdapter):
    name = "mathqa"
    default_system_prompt = MATH_SYSTEM_PROMPT

    def load_records(self, spec: DatasetSpec, limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        local_path = find_first_existing_path(spec.local_paths)
        if local_path is not None:
            if local_path.is_file():
                return self._load_json_file(local_path, limit=limit)
            return self._load_saved_dataset(local_path, split=spec.split, limit=limit)
        if spec.hf_dataset_id:
            return self._load_hf_dataset(spec, limit=limit)
        raise FileNotFoundError(f"Khong tim thay dataset cho {spec.name}")

    def prepare_sample(self, record: dict[str, Any], index: int) -> PreparedSample:
        problem = _first_non_empty_value(record, ["Problem", "problem", "question"])
        options = _first_non_empty_value(record, ["options", "Options"])
        correct_answer = _first_non_empty_value(record, ["correct", "answer"])
        options_text = _stringify_options(options)
        user_content = f"{problem}\n\nOptions:\n{options_text}"
        sample_id = str(record.get("id") or record.get("idx") or index)
        return PreparedSample(
            sample_id=sample_id,
            messages=[
                {"role": "system", "content": self.default_system_prompt},
                {"role": "user", "content": user_content},
            ],
            reference_text=str(correct_answer),
            reference_answer=str(correct_answer).strip().lower(),
            raw_sample=json_ready(record),
            metadata={
                "problem": problem,
                "options": options_text,
            },
        )

    def extract_prediction(self, generated_text: str, sample: PreparedSample) -> str:
        return extract_mathqa_choice(generated_text)

    def score_sample(
        self,
        sample: PreparedSample,
        predicted_answer: str,
        generated_text: str,
    ) -> dict[str, Any]:
        is_correct, detail = compare_choice_answers(predicted_answer, sample.reference_answer)
        return {
            "metric_name": "accuracy",
            "metric_value": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "detail": detail,
        }


def _first_non_empty_value(record: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value
        if not isinstance(value, str):
            return str(value)
    raise KeyError(f"Khong tim thay key nao trong {keys}")


def _extract_user_prompt(record: dict[str, Any]) -> str:
    for field_name in ["messages", "conversations"]:
        if field_name in record:
            messages = record[field_name]
            user_content = _find_message_content(messages, {"user", "human"})
            if user_content:
                return user_content

    instruction = record.get("instruction")
    input_text = record.get("input")
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    if instruction:
        return str(instruction)

    return _first_non_empty_value(record, ["prompt", "question", "problem", "Problem", "query"])


def _extract_assistant_text(record: dict[str, Any]) -> str:
    for field_name in ["messages", "conversations"]:
        if field_name in record:
            messages = record[field_name]
            assistant_content = _find_message_content(messages, {"assistant", "gpt"}, reverse=True)
            if assistant_content:
                return assistant_content

    return _first_non_empty_value(record, ["output", "response", "answer", "solution"])


def _find_message_content(
    messages: Any,
    accepted_roles: set[str],
    reverse: bool = False,
) -> str | None:
    if not isinstance(messages, list):
        return None

    iterable = reversed(messages) if reverse else messages
    for message in iterable:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or message.get("from") or message.get("speaker") or "").lower()
        if role not in accepted_roles:
            continue
        content = message.get("content") or message.get("value") or message.get("text")
        if content:
            return str(content)
    return None


def _stringify_options(options: Any) -> str:
    if isinstance(options, str):
        return options
    if isinstance(options, list):
        return "\n".join(str(item) for item in options)
    return str(options)


DATASET_ADAPTERS: dict[str, BaseDatasetAdapter] = {
    AceReasonAdapter.name: AceReasonAdapter(),
    GSM8KAdapter.name: GSM8KAdapter(),
    MathQAAdapter.name: MathQAAdapter(),
}


def get_dataset_adapter(name: str) -> BaseDatasetAdapter:
    if name not in DATASET_ADAPTERS:
        raise KeyError(f"Dataset adapter '{name}' chua duoc dang ky.")
    return DATASET_ADAPTERS[name]
