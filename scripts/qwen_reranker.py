"""Traceable single-GPU Qwen3 cross-encoder reranker."""

from __future__ import annotations

import time
from typing import Any


DEFAULT_INSTRUCTION = (
    "Given a user question, retrieve passages that contain sufficient evidence "
    "to answer the question"
)
SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)


def format_instruction(instruction: str, query: str, document: str) -> str:
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


def rank_scored_candidates(
    candidates: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join model scores to candidates and apply a deterministic ranking."""
    if len(candidates) != len(score_rows):
        raise ValueError("Candidates and score rows must have equal length")
    if not candidates:
        raise ValueError("At least one candidate is required")

    joined = []
    for candidate, score_row in zip(candidates, score_rows):
        joined.append(
            {
                **candidate,
                "hybrid_rank": int(candidate["rank"]),
                "hybrid_score": candidate["score"],
                "reranker_logit_difference": float(
                    score_row["reranker_logit_difference"]
                ),
                "relevance_probability": float(
                    score_row["relevance_probability"]
                ),
                "input_tokens": int(score_row["input_tokens"]),
            }
        )

    joined.sort(
        key=lambda item: (
            -item["reranker_logit_difference"],
            item["hybrid_rank"],
            item["chunk_id"],
        )
    )
    results = []
    for rank, item in enumerate(joined, start=1):
        result = dict(item)
        result["rank"] = rank
        result["score"] = round(item["reranker_logit_difference"], 6)
        result["reranker_logit_difference"] = round(
            item["reranker_logit_difference"],
            6,
        )
        result["relevance_probability"] = round(
            item["relevance_probability"],
            9,
        )
        results.append(result)
    return results


class QwenReranker:
    """Score Query–Chunk pairs from the final yes/no token logits."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        instruction: str = DEFAULT_INSTRUCTION,
        max_length: int = 8192,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.model_id = model_id
        self.revision = revision
        self.instruction = instruction
        self.max_length = max_length

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        model_load_started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        yes_ids = self.tokenizer("yes", add_special_tokens=False).input_ids
        no_ids = self.tokenizer("no", add_special_tokens=False).input_ids
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise AssertionError("yes and no must each map to exactly one token")
        self.yes_token_id = yes_ids[0]
        self.no_token_id = no_ids[0]

        self.model_prefix = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.model_suffix = (
            "<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        self.prefix_tokens = self.tokenizer.encode(
            self.model_prefix,
            add_special_tokens=False,
        )
        self.suffix_tokens = self.tokenizer.encode(
            self.model_suffix,
            add_special_tokens=False,
        )
        available_pair_tokens = (
            self.max_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens)
        )
        if available_pair_tokens <= 0:
            raise ValueError("max_length is too small for the reranker template")
        self.available_pair_tokens = available_pair_tokens
        self.model_load_seconds = time.perf_counter() - model_load_started

    def _prepare_inputs(
        self,
        query: str,
        documents: list[str],
    ) -> tuple[dict[str, Any], list[str]]:
        formatted_pairs = [
            format_instruction(self.instruction, query, document)
            for document in documents
        ]
        tokenized = self.tokenizer(
            formatted_pairs,
            padding=False,
            truncation=True,
            max_length=self.available_pair_tokens,
            return_attention_mask=False,
        )
        input_ids = [
            self.prefix_tokens + pair_ids + self.suffix_tokens
            for pair_ids in tokenized["input_ids"]
        ]
        model_inputs = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            key: value.to(self.model.device)
            for key, value in model_inputs.items()
        }, formatted_pairs

    def score(
        self,
        query: str,
        documents: list[str],
    ) -> dict[str, Any]:
        if not documents:
            raise ValueError("At least one document is required")
        model_inputs, formatted_pairs = self._prepare_inputs(query, documents)
        input_token_counts = model_inputs["attention_mask"].sum(dim=1).tolist()

        self.torch.cuda.reset_peak_memory_stats()
        self.torch.cuda.synchronize()
        scoring_started = time.perf_counter()
        with self.torch.inference_mode():
            final_logits = self.model(**model_inputs).logits[:, -1, :]
            yes_logits = final_logits[:, self.yes_token_id].float()
            no_logits = final_logits[:, self.no_token_id].float()
            binary_logits = self.torch.stack([no_logits, yes_logits], dim=1)
            relevance_probabilities = self.torch.softmax(binary_logits, dim=1)[:, 1]
            logit_differences = yes_logits - no_logits
        self.torch.cuda.synchronize()
        scoring_seconds = time.perf_counter() - scoring_started

        scores = [
            {
                "reranker_logit_difference": float(logit_difference),
                "relevance_probability": float(probability),
                "input_tokens": int(input_token_count),
            }
            for logit_difference, probability, input_token_count in zip(
                logit_differences.cpu(),
                relevance_probabilities.cpu(),
                input_token_counts,
            )
        ]
        return {
            "formatted_pairs": formatted_pairs,
            "scores": scores,
            "scoring_seconds": round(scoring_seconds, 6),
            "peak_gpu_memory_gib": round(
                self.torch.cuda.max_memory_allocated() / (1024**3),
                3,
            ),
        }

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        scored = self.score(
            query,
            [candidate["text"] for candidate in candidates],
        )
        results = rank_scored_candidates(candidates, scored["scores"])
        return {
            "query": query,
            "instruction": self.instruction,
            "system_prompt": SYSTEM_PROMPT,
            "model_prefix": self.model_prefix,
            "model_suffix": self.model_suffix,
            "formatted_pairs": scored["formatted_pairs"],
            "retrieved_ids": [result["chunk_id"] for result in results],
            "retrieval_scores": [result["score"] for result in results],
            "results": results,
            "latency_seconds": {
                "reranker_scoring": scored["scoring_seconds"],
            },
            "peak_gpu_memory_gib": scored["peak_gpu_memory_gib"],
        }
