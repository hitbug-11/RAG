"""Run the q01 Top-1/Top-2 Generator probe with pinned Qwen3-8B."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import accelerate
import torch
import transformers

from qwen_generator import QwenGenerator


MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalized_text(text: str) -> str:
    return "".join(text.split()).lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, default=Path("results/day1_context_packing.jsonl"))
    parser.add_argument("--retrieval-traces", type=Path, default=Path("results/day1_dense_retrieval.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/eval/day1_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/day1_q01_generator_probe.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/day1_q01_generator_probe_summary.json"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")

    contexts = [
        record
        for record in read_jsonl(args.contexts)
        if record["question_id"] == "q01" and record["condition"] in {"top_1", "top_2"}
    ]
    contexts.sort(key=lambda record: record["requested_top_k"])
    if [record["condition"] for record in contexts] != ["top_1", "top_2"]:
        raise AssertionError("Expected exactly q01 top_1 and top_2 context records")

    questions = {record["question_id"]: record for record in read_jsonl(args.questions)}
    question = questions["q01"]
    aliases = [question["expected_answer"], *question.get("answer_aliases", [])]
    normalized_aliases = {normalized_text(alias) for alias in aliases}

    retrieval_trace = next(
        record for record in read_jsonl(args.retrieval_traces) if record["question_id"] == "q01"
    )
    chunks_by_id = {result["chunk_id"]: result for result in retrieval_trace["results"]}

    generator = QwenGenerator(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        max_new_tokens=args.max_new_tokens,
    )
    traces: list[dict[str, Any]] = []
    for context_record in contexts:
        generation = generator.generate(context_record["messages"], seed=42)
        parsed = generation["parsed_output"]
        answer = parsed.get("answer") if isinstance(parsed, dict) else None
        citations = parsed.get("citations") if isinstance(parsed, dict) else None
        insufficient_evidence = parsed.get("insufficient_evidence") if isinstance(parsed, dict) else None

        citations_are_list = isinstance(citations, list) and all(isinstance(item, str) for item in citations)
        selected_ids = set(context_record["packing"]["selected_ids"])
        citation_ids_valid = citations_are_list and set(citations).issubset(selected_ids)
        answer_matches_expected = isinstance(answer, str) and any(
            alias in normalized_text(answer) for alias in normalized_aliases
        )
        citation_supports_expected_answer = citations_are_list and any(
            citation in chunks_by_id
            and any(alias in normalized_text(chunks_by_id[citation]["text"]) for alias in normalized_aliases)
            for citation in citations
        )

        if context_record["evidence_present_in_prompt"]:
            expected_behavior_met = (
                answer_matches_expected
                and insufficient_evidence is False
                and citation_ids_valid
                and citation_supports_expected_answer
            )
        else:
            expected_behavior_met = (
                insufficient_evidence is True
                and citations_are_list
                and len(citations) == 0
                and not answer_matches_expected
            )

        traces.append(
            {
                "question_id": "q01",
                "condition": context_record["condition"],
                "query": context_record["query"],
                "expected_answer": context_record["expected_answer"],
                "evidence_present_in_prompt": context_record["evidence_present_in_prompt"],
                "selected_ids": context_record["packing"]["selected_ids"],
                "retrieval_scores": context_record["packing"]["selected_scores"],
                "packed_context": context_record["packing"]["packed_context"],
                "messages": context_record["messages"],
                **generation,
                "evaluation": {
                    "answer_matches_expected": answer_matches_expected,
                    "insufficient_evidence": insufficient_evidence,
                    "citation_ids_valid": citation_ids_valid,
                    "citation_supports_expected_answer": citation_supports_expected_answer,
                    "expected_behavior_met": expected_behavior_met,
                },
            }
        )

    write_jsonl(args.output, traces)
    summary = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_load_seconds": round(generator.model_load_seconds, 3),
        "conditions": [
            {
                "condition": trace["condition"],
                "evidence_present_in_prompt": trace["evidence_present_in_prompt"],
                "raw_output": trace["raw_output"],
                "parsed_output": trace["parsed_output"],
                "parse_error": trace["parse_error"],
                "evaluation": trace["evaluation"],
                "token_usage": trace["token_usage"],
                "generation_seconds": trace["generation_seconds"],
                "peak_gpu_memory_gib": trace["peak_gpu_memory_gib"],
            }
            for trace in traces
        ],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "visible_gpu_count": torch.cuda.device_count(),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
