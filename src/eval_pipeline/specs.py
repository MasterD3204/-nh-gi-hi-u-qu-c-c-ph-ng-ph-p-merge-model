from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ModelSpec:
    name: str
    local_paths: list[str] = field(default_factory=list)
    hf_repo_id: str | None = None
    revision: str | None = None
    tokenizer_name: str | None = None
    trust_remote_code: bool = False
    torch_dtype: str = "auto"
    device_map: str = "auto"
    attn_implementation: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSpec:
    name: str
    adapter_name: str
    local_paths: list[str] = field(default_factory=list)
    hf_dataset_id: str | None = None
    hf_subset: str | None = None
    split: str = "test"
    revision: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationSettings:
    batch_size: int = 1
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    num_beams: int = 1
    use_cache: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreparedSample:
    sample_id: str
    messages: list[dict[str, str]]
    reference_text: str
    reference_answer: str
    raw_sample: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationRequest:
    model_names: list[str]
    dataset_names: list[str]
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    seed: int = 42
    run_name: str | None = None
    output_root: str = "outputs/runs"
    default_dataset_limit: int | None = None
    per_dataset_limit: dict[str, int] = field(default_factory=dict)
    system_prompt_overrides: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["generation"] = self.generation.to_dict()
        return payload
