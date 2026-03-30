from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .specs import GenerationSettings, ModelSpec
from .utils import resolve_artifact_source


def _resolve_torch_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


class TransformersChatRunner:
    def __init__(self, model_spec: ModelSpec) -> None:
        self.model_spec = model_spec
        self.source = resolve_artifact_source(model_spec.local_paths, model_spec.hf_repo_id)
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self) -> "TransformersChatRunner":
        source_value = self.source["value"]
        tokenizer_source = self.model_spec.tokenizer_name or source_value

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            revision=self.model_spec.revision,
            trust_remote_code=self.model_spec.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "revision": self.model_spec.revision,
            "trust_remote_code": self.model_spec.trust_remote_code,
            "torch_dtype": _resolve_torch_dtype(self.model_spec.torch_dtype),
            "device_map": self.model_spec.device_map,
        }
        if self.model_spec.attn_implementation:
            model_kwargs["attn_implementation"] = self.model_spec.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(source_value, **model_kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        return self

    def generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        generation: GenerationSettings,
    ) -> list[dict[str, str]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model va tokenizer chua duoc load.")

        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_batch
        ]

        tokenized_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        tokenized_inputs = {key: value.to(self.device) for key, value in tokenized_inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": generation.max_new_tokens,
            "do_sample": generation.do_sample,
            "num_beams": generation.num_beams,
            "repetition_penalty": generation.repetition_penalty,
            "use_cache": generation.use_cache,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if generation.temperature is not None:
            generate_kwargs["temperature"] = generation.temperature
        if generation.top_p is not None:
            generate_kwargs["top_p"] = generation.top_p

        with torch.inference_mode():
            outputs = self.model.generate(**tokenized_inputs, **generate_kwargs)

        prompt_length = tokenized_inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_length:]
        decoded_outputs = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return [
            {
                "rendered_prompt": prompt,
                "generated_text": decoded_output.strip(),
            }
            for prompt, decoded_output in zip(prompts, decoded_outputs)
        ]

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def describe(self) -> dict[str, Any]:
        return {
            "model_name": self.model_spec.name,
            "resolved_source": self.source,
            "tokenizer_name": self.model_spec.tokenizer_name or self.source["value"],
            "device": str(self.device) if self.device is not None else None,
            "torch_dtype": self.model_spec.torch_dtype,
            "device_map": self.model_spec.device_map,
            "trust_remote_code": self.model_spec.trust_remote_code,
            "revision": self.model_spec.revision,
        }

