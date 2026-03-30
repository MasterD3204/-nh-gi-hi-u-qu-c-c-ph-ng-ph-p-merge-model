from __future__ import annotations

from dataclasses import dataclass

from .specs import DatasetSpec, ModelSpec


@dataclass
class ExperimentRegistry:
    models: dict[str, ModelSpec]
    datasets: dict[str, DatasetSpec]


def build_default_registry() -> ExperimentRegistry:
    models = {
        "qwen3_mathqa_merged": ModelSpec(
            name="qwen3_mathqa_merged",
            local_paths=[
                "/content/drive/MyDrive/Khoá_luận_tốt_nghiệp/Model/Qwen3-1.7B-mathqa-merged",
            ],
        ),
        "qwen3_gsm8k_merged": ModelSpec(
            name="qwen3_gsm8k_merged",
            local_paths=[
                "/content/drive/MyDrive/Khoá_luận_tốt_nghiệp/Model/qwen3-1.7b-gsm8k-merged",
            ],
        ),
        "qwen3_acereason_merged": ModelSpec(
            name="qwen3_acereason_merged",
            local_paths=[
                "/content/drive/MyDrive/Khoá_luận_tốt_nghiệp/Model/Qwen3-1.7B-acereason-merged",
            ],
        ),
    }

    datasets = {
        "acereason_test": DatasetSpec(
            name="acereason_test",
            adapter_name="ace_reason",
            local_paths=[
                "/content/drive/MyDrive/Khoá_luận_tốt_nghiệp/Dataset/AceReason-1.1-SFT-Filtered",
            ],
            split="test",
        ),
        "gsm8k_test": DatasetSpec(
            name="gsm8k_test",
            adapter_name="gsm8k",
            local_paths=[
                "/content/drive/MyDrive/Khoá_luận_tốt_nghiệp/Dataset/gsm8k",
            ],
            hf_dataset_id="gsm8k",
            hf_subset="main",
            split="test",
        ),
        "mathqa_test": DatasetSpec(
            name="mathqa_test",
            adapter_name="mathqa",
            local_paths=[
                "/content/drive/MyDrive/Khoá_luận_tốt_nghiệp/Dataset/mathqa/test.json",
            ],
            hf_dataset_id="math_qa",
            split="test",
        ),
    }

    return ExperimentRegistry(models=models, datasets=datasets)
