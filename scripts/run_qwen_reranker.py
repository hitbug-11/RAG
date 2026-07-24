"""Rerank Day 2 RRF Hybrid candidates with pinned Qwen3-Reranker."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

import torch
import transformers

from qwen_reranker import DEFAULT_INSTRUCTION, QwenReranker


MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "5340c0261aa49a842d1bff01db91ce407bda87a2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_text(text: str) -> str:
    return "".join(text.split()).lower()


def first_matching_rank(
    results: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> int | None:
    for result in results:
        if predicate(result):
            return int(result["rank"])
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/eval/day1_questions.jsonl"))
    parser.add_argument(
        "--hybrid-traces",
        type=Path,
        default=Path("results/day2_rrf_hybrid_retrieval.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/day2_qwen_reranker.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/day2_qwen_reranker_summary.json"),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("results/day2_reranker_comparison.csv"),
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--candidate-depth", type=int, default=30)
    parser.add_argument("--output-top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this server experiment")
    if args.candidate_depth <= 0 or args.output_top_k <= 0:
        raise ValueError("candidate-depth and output-top-k must be positive")

    questions = read_jsonl(args.questions)
    hybrid_by_question = {
        trace["question_id"]: trace
        for trace in read_jsonl(args.hybrid_traces)
    }
    if {question["question_id"] for question in questions} != set(hybrid_by_question):
        raise ValueError("Questions and Hybrid traces must contain the same IDs")

    reranker = QwenReranker(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        instruction=DEFAULT_INSTRUCTION,
        max_length=args.max_length,
    )
    experiment_started = time.perf_counter()
    traces: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for question in questions:
        question_id = question["question_id"]
        hybrid_trace = hybrid_by_question[question_id]
        candidate_results = hybrid_trace["results"][: args.candidate_depth]
        scored_trace = reranker.rerank(
            question["query"],
            candidate_results,
        )
        all_scored_results = scored_trace["results"]
        output_results = all_scored_results[: args.output_top_k]
        trace = {
            **scored_trace,
            "candidate_count": len(candidate_results),
            "output_top_k": args.output_top_k,
            "all_scored_results": all_scored_results,
            "results": output_results,
            "retrieved_ids": [
                result["chunk_id"] for result in output_results
            ],
            "retrieval_scores": [
                result["score"] for result in output_results
            ],
        }
        aliases = [question["expected_answer"], *question.get("answer_aliases", [])]
        normalized_aliases = {normalized_text(alias) for alias in aliases}
        document_rank = first_matching_rank(
            trace["results"],
            lambda result: result["document_id"] == question["gold_document_id"],
        )
        answer_rank = first_matching_rank(
            trace["results"],
            lambda result: result["document_id"] == question["gold_document_id"]
            and any(alias in normalized_text(result["text"]) for alias in normalized_aliases),
        )
        trace.update(
            {
                "question_id": question_id,
                "expected_answer": question["expected_answer"],
                "gold_document_id": question["gold_document_id"],
                "gold_document_best_rank": document_rank,
                "gold_answer_best_rank": answer_rank,
                "document_hit_at_1": document_rank == 1,
                "document_hit_at_3": document_rank is not None and document_rank <= 3,
                "answer_hit_at_1": answer_rank == 1,
                "answer_hit_at_3": answer_rank is not None and answer_rank <= 3,
            }
        )
        traces.append(trace)

        hybrid_answer_rank = hybrid_trace["gold_answer_best_rank"]
        comparison_rows.append(
            {
                "question_id": question_id,
                "hybrid_top1_chunk_id": hybrid_trace["retrieved_ids"][0],
                "reranker_top1_chunk_id": trace["retrieved_ids"][0],
                "top1_changed": (
                    hybrid_trace["retrieved_ids"][0] != trace["retrieved_ids"][0]
                ),
                "hybrid_gold_answer_rank": hybrid_answer_rank,
                "reranker_gold_answer_rank": answer_rank,
                "answer_rank_improvement": (
                    None
                    if hybrid_answer_rank is None or answer_rank is None
                    else hybrid_answer_rank - answer_rank
                ),
                "reranker_top1_score": trace["retrieval_scores"][0],
                "reranker_top1_probability": trace["results"][0][
                    "relevance_probability"
                ],
                "reranker_top1_logit_margin": round(
                    trace["results"][0]["reranker_logit_difference"]
                    - trace["results"][1]["reranker_logit_difference"],
                    6,
                ),
                "reranker_top1_hybrid_rank": trace["results"][0]["hybrid_rank"],
            }
        )

    experiment_seconds = time.perf_counter() - experiment_started
    write_jsonl(args.output, traces)
    write_csv(args.comparison, comparison_rows)

    question_count = len(traces)
    document_ranks = [trace["gold_document_best_rank"] for trace in traces]
    answer_ranks = [trace["gold_answer_best_rank"] for trace in traces]
    summary = {
        "retriever": "BM25 + Dense RRF + Qwen3 Reranker",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "instruction": DEFAULT_INSTRUCTION,
        "parameters": {
            "requested_candidate_depth": args.candidate_depth,
            "actual_candidate_depths": sorted(
                {trace["candidate_count"] for trace in traces}
            ),
            "output_top_k": args.output_top_k,
            "max_length": args.max_length,
            "dtype": "bfloat16",
        },
        "metrics": {
            "question_count": question_count,
            "document_recall_at_1": sum(rank == 1 for rank in document_ranks) / question_count,
            "document_recall_at_3": sum(
                rank is not None and rank <= 3 for rank in document_ranks
            )
            / question_count,
            "answer_recall_at_1": sum(rank == 1 for rank in answer_ranks) / question_count,
            "answer_recall_at_3": sum(rank is not None and rank <= 3 for rank in answer_ranks)
            / question_count,
            "document_mrr": round(
                sum(reciprocal_rank(rank) for rank in document_ranks) / question_count,
                6,
            ),
            "answer_mrr": round(
                sum(reciprocal_rank(rank) for rank in answer_ranks) / question_count,
                6,
            ),
            "top1_changed_question_count": sum(
                row["top1_changed"] for row in comparison_rows
            ),
            "answer_rank_improved_question_count": sum(
                (row["answer_rank_improvement"] or 0) > 0
                for row in comparison_rows
            ),
        },
        "per_question": [
            {
                "question_id": trace["question_id"],
                "gold_answer_best_rank": trace["gold_answer_best_rank"],
                "top1_chunk_id": trace["retrieved_ids"][0],
                "top1_logit_difference": trace["retrieval_scores"][0],
                "top1_relevance_probability": trace["results"][0][
                    "relevance_probability"
                ],
                "top1_logit_margin": round(
                    trace["results"][0]["reranker_logit_difference"]
                    - trace["results"][1]["reranker_logit_difference"],
                    6,
                ),
                "top1_hybrid_rank": trace["results"][0]["hybrid_rank"],
                "scoring_seconds": trace["latency_seconds"]["reranker_scoring"],
                "peak_gpu_memory_gib": trace["peak_gpu_memory_gib"],
            }
            for trace in traces
        ],
        "timing": {
            "model_load_seconds": round(reranker.model_load_seconds, 3),
            "five_question_experiment_seconds": round(experiment_seconds, 3),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "artifacts": {
            "traces": str(args.output),
            "summary": str(args.summary),
            "comparison": str(args.comparison),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
