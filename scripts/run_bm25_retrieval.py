"""Build BM25 and compare it with the recorded Day 1 dense retrieval baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

from bm25_retriever import BM25Retriever


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


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path("results/day1_chunks.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/eval/day1_questions.jsonl"))
    parser.add_argument(
        "--dense-traces",
        type=Path,
        default=Path("results/day1_dense_retrieval.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/day2_bm25_retrieval.jsonl"))
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/day2_bm25_retrieval_summary.json"),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("results/day2_bm25_dense_comparison.csv"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = read_jsonl(args.chunks)
    questions = read_jsonl(args.questions)
    dense_by_question = {
        trace["question_id"]: trace for trace in read_jsonl(args.dense_traces)
    }
    if {question["question_id"] for question in questions} != set(dense_by_question):
        raise ValueError("Dense traces and evaluation questions do not contain the same IDs")

    build_started = time.perf_counter()
    retriever = BM25Retriever(chunks, k1=args.k1, b=args.b)
    build_seconds = time.perf_counter() - build_started

    traces: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for question in questions:
        trace = retriever.search(question["query"], top_k=args.top_k)
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
                "question_id": question["question_id"],
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

        dense_trace = dense_by_question[question["question_id"]]
        comparison_rows.append(
            {
                "question_id": question["question_id"],
                "bm25_top1_chunk_id": trace["retrieved_ids"][0],
                "dense_top1_chunk_id": dense_trace["retrieved_ids"][0],
                "top1_same_chunk": trace["retrieved_ids"][0] == dense_trace["retrieved_ids"][0],
                "bm25_gold_document_rank": document_rank,
                "dense_gold_document_rank": dense_trace["gold_document_best_rank"],
                "bm25_gold_answer_rank": answer_rank,
                "dense_gold_answer_rank": dense_trace["gold_answer_best_rank"],
                "bm25_top1_score": trace["retrieval_scores"][0],
                "dense_top1_score": dense_trace["retrieval_scores"][0],
            }
        )

    write_jsonl(args.output, traces)
    write_comparison_csv(args.comparison, comparison_rows)

    question_count = len(traces)
    document_ranks = [trace["gold_document_best_rank"] for trace in traces]
    answer_ranks = [trace["gold_answer_best_rank"] for trace in traces]
    summary = {
        "retriever": "BM25",
        "tokenizer": "NFKC lowercase; Chinese character unigrams+bigrams; alphanumeric spans",
        "parameters": {"k1": args.k1, "b": args.b},
        "chunks_sha256": hashlib.sha256(args.chunks.read_bytes()).hexdigest(),
        "chunk_count": len(chunks),
        "average_document_length": round(retriever.average_document_length, 3),
        "vocabulary_size": len(retriever.document_frequencies),
        "top_k": args.top_k,
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
            "top1_chunk_agreement_with_dense": sum(
                row["top1_same_chunk"] for row in comparison_rows
            )
            / question_count,
        },
        "per_question": [
            {
                "question_id": trace["question_id"],
                "gold_document_best_rank": trace["gold_document_best_rank"],
                "gold_answer_best_rank": trace["gold_answer_best_rank"],
                "top1_chunk_id": trace["retrieved_ids"][0],
                "top1_score": trace["retrieval_scores"][0],
                "top1_strongest_terms": [
                    item["term"] for item in trace["results"][0]["top_term_contributions"][:5]
                ],
            }
            for trace in traces
        ],
        "timing": {"index_build_seconds": round(build_seconds, 6)},
        "environment": {"python": platform.python_version()},
        "artifacts": {
            "traces": str(args.output),
            "summary": str(args.summary),
            "dense_comparison": str(args.comparison),
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
