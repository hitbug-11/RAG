"""Compare Top-1 and Top-2 Context Packing for the Day 1 retrieval traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from context_pipeline import ContextPacker, build_prompt


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
    parser.add_argument("--retrieval-traces", type=Path, default=Path("results/day1_dense_retrieval.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/eval/day1_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/day1_context_packing.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/day1_context_packing_summary.json"))
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--max-context-chars", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_traces = read_jsonl(args.retrieval_traces)
    questions = {item["question_id"]: item for item in read_jsonl(args.questions)}
    packer = ContextPacker(max_context_chars=args.max_context_chars)
    records: list[dict[str, Any]] = []

    for retrieval_trace in retrieval_traces:
        question = questions[retrieval_trace["question_id"]]
        aliases = [question["expected_answer"], *question.get("answer_aliases", [])]
        normalized_aliases = {normalized_text(alias) for alias in aliases}

        for top_k in args.top_k:
            requested_results = retrieval_trace["results"][:top_k]
            packed = packer.pack(requested_results)
            prompt_data = build_prompt(question["query"], packed)
            evidence_present = any(
                alias in normalized_text(packed.packed_context) for alias in normalized_aliases
            )

            # Every selected ID and complete Chunk text must be visible in the prompt.
            for result in requested_results[: len(packed.selected_ids)]:
                if result["chunk_id"] not in prompt_data["prompt"]:
                    raise AssertionError(f"Chunk ID missing from prompt: {result['chunk_id']}")
                if result["text"] not in prompt_data["prompt"]:
                    raise AssertionError(f"Chunk text was altered or truncated: {result['chunk_id']}")

            records.append(
                {
                    "question_id": question["question_id"],
                    "query": question["query"],
                    "expected_answer": question["expected_answer"],
                    "condition": f"top_{top_k}",
                    "requested_top_k": top_k,
                    "retrieved_ids": retrieval_trace["retrieved_ids"][:top_k],
                    "retrieval_scores": retrieval_trace["retrieval_scores"][:top_k],
                    "evidence_present_in_prompt": evidence_present,
                    **prompt_data,
                }
            )

    write_jsonl(args.output, records)
    summary_by_top_k: dict[str, Any] = {}
    for top_k in args.top_k:
        subset = [record for record in records if record["requested_top_k"] == top_k]
        summary_by_top_k[f"top_{top_k}"] = {
            "record_count": len(subset),
            "evidence_coverage": sum(record["evidence_present_in_prompt"] for record in subset) / len(subset),
            "mean_context_chars": round(
                sum(record["packing"]["context_char_count"] for record in subset) / len(subset), 2
            ),
            "mean_prompt_chars": round(
                sum(record["prompt_char_count"] for record in subset) / len(subset), 2
            ),
            "dropped_chunk_count": sum(len(record["packing"]["dropped_ids"]) for record in subset),
        }

    q01_records = [record for record in records if record["question_id"] == "q01"]
    summary = {
        "retrieval_trace_count": len(retrieval_traces),
        "prompt_record_count": len(records),
        "max_context_chars": args.max_context_chars,
        "by_top_k": summary_by_top_k,
        "q01_propagation": [
            {
                "condition": record["condition"],
                "selected_ids": record["packing"]["selected_ids"],
                "evidence_present_in_prompt": record["evidence_present_in_prompt"],
                "context_char_count": record["packing"]["context_char_count"],
            }
            for record in q01_records
        ],
        "validation": {
            "all_selected_chunk_ids_preserved": True,
            "all_selected_chunk_text_preserved": True,
            "no_chunk_dropped_by_current_budget": all(
                not record["packing"]["dropped_ids"] for record in records
            ),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
