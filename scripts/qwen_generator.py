"""Traceable single-GPU Qwen3 generator for controlled RAG experiments."""

from __future__ import annotations

import json
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def parse_json_output(raw_output: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the first outer JSON object while preserving failures in the trace."""
    stripped = raw_output.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        return None, "No JSON object found"
    candidate = stripped[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as error:
        return None, f"JSONDecodeError: {error}"
    if not isinstance(parsed, dict):
        return None, "Parsed JSON is not an object"
    return parsed, None


class QwenGenerator:
    """Load one pinned Qwen3 model and retain the exact model-side prompt."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens

        load_started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            bos_token_id=self.model.generation_config.bos_token_id,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.model_load_seconds = time.perf_counter() - load_started

    def generate(self, messages: list[dict[str, str]], *, seed: int = 42) -> dict[str, Any]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        rendered_chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([rendered_chat_prompt], return_tensors="pt").to(self.model.device)
        input_token_count = int(model_inputs.input_ids.shape[1])

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config,
                use_model_defaults=False,
                use_cache=True,
            )
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started

        output_ids = generated_ids[0, input_token_count:]
        raw_output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        parsed_output, parse_error = parse_json_output(raw_output)

        return {
            "rendered_chat_prompt": rendered_chat_prompt,
            "raw_output": raw_output,
            "parsed_output": parsed_output,
            "parse_error": parse_error,
            "token_usage": {
                "input_tokens": input_token_count,
                "output_tokens": int(output_ids.shape[0]),
            },
            "generation_seconds": round(generation_seconds, 3),
            "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
            "generation_config": {
                "enable_thinking": False,
                "do_sample": False,
                "use_model_defaults": False,
                "max_new_tokens": self.max_new_tokens,
                "seed": seed,
                "dtype": "bfloat16",
            },
        }
