"""Fuse recorded BM25 and Dense rankings with transparent RRF."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Any, Callable

from rrf_fusion import reciprocal_rank_fusion


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
        "--bm25-traces",
        type=Path,
        default=Path("results/day2_bm25_retrieval.jsonl"),
    )
    parser.add_argument(
        "--dense-traces",
        type=Path,
        default=Path("results/day1_dense_retrieval.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/day2_rrf_hybrid_retrieval.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/day2_rrf_hybrid_retrieval_summary.json"),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("results/day2_rrf_retriever_comparison.csv"),
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = read_jsonl(args.questions)
    bm25_by_question = {
        trace["question_id"]: trace for trace in read_jsonl(args.bm25_traces)
    }
    dense_by_question = {
        trace["question_id"]: trace for trace in read_jsonl(args.dense_traces)
    }
    question_ids = {question["question_id"] for question in questions}
    if question_ids != set(bm25_by_question) or question_ids != set(dense_by_question):
        raise ValueError("Questions, BM25 traces, and Dense traces must contain the same IDs")

    traces: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for question in questions:
        question_id = question["question_id"]
        bm25_trace = bm25_by_question[question_id]
        dense_trace = dense_by_question[question_id]
        trace = reciprocal_rank_fusion(
            {
                "bm25": bm25_trace["results"],
                "dense": dense_trace["results"],
            },
            rrf_k=args.rrf_k,
            top_k=args.top_k,
        )
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
                "query": question["query"],
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

        comparison_rows.append(
            {
                "question_id": question_id,
                "bm25_top1_chunk_id": bm25_trace["retrieved_ids"][0],
                "dense_top1_chunk_id": dense_trace["retrieved_ids"][0],
                "hybrid_top1_chunk_id": trace["retrieved_ids"][0],
                "bm25_gold_answer_rank": bm25_trace["gold_answer_best_rank"],
                "dense_gold_answer_rank": dense_trace["gold_answer_best_rank"],
                "hybrid_gold_answer_rank": answer_rank,
                "hybrid_top1_bm25_rank": trace["results"][0]["source_ranks"].get("bm25"),
                "hybrid_top1_dense_rank": trace["results"][0]["source_ranks"].get("dense"),
                "hybrid_top1_score": trace["retrieval_scores"][0],
                "top_score_tied": trace["top_score_tied"],
                "candidate_union_size": trace["candidate_union_size"],
            }
        )

    write_jsonl(args.output, traces)
    write_csv(args.comparison, comparison_rows)

    question_count = len(traces)
    document_ranks = [trace["gold_document_best_rank"] for trace in traces]
    answer_ranks = [trace["gold_answer_best_rank"] for trace in traces]
    top_score_tie_ids = [
        trace["question_id"] for trace in traces if trace["top_score_tied"]
    ]
    summary = {
        "retriever": "BM25 + Dense RRF",
        "parameters": {
            "rrf_k": args.rrf_k,
            "source_depth": 5,
            "output_top_k": args.top_k,
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
            "top1_agreement_with_bm25": sum(
                trace["retrieved_ids"][0]
                == bm25_by_question[trace["question_id"]]["retrieved_ids"][0]
                for trace in traces
            )
            / question_count,
            "top1_agreement_with_dense": sum(
                trace["retrieved_ids"][0]
                == dense_by_question[trace["question_id"]]["retrieved_ids"][0]
                for trace in traces
            )
            / question_count,
            "top_score_tie_question_count": len(top_score_tie_ids),
            "top_score_tie_question_ids": top_score_tie_ids,
        },
        "per_question": [
            {
                "question_id": trace["question_id"],
                "gold_answer_best_rank": trace["gold_answer_best_rank"],
                "top1_chunk_id": trace["retrieved_ids"][0],
                "top1_score": trace["retrieval_scores"][0],
                "top1_source_ranks": trace["results"][0]["source_ranks"],
                "top_score_tied": trace["top_score_tied"],
                "candidate_union_size": trace["candidate_union_size"],
            }
            for trace in traces
        ],
        "environment": {"python": platform.python_version()},
        "artifacts": {
            "traces": str(args.output),
            "summary": str(args.summary),
            "retriever_comparison": str(args.comparison),
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
