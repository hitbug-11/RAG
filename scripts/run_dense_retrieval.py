"""Build and evaluate the Day 1 dense retrieval baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import sentence_transformers
import torch
import transformers

from dense_retriever import DenseRetriever, read_jsonl


MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def normalized_text(text: str) -> str:
    return "".join(text.split()).lower()


def first_matching_rank(results: list[dict[str, Any]], predicate: Any) -> int | None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path("results/day1_chunks.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/eval/day1_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/day1_dense_retrieval.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/day1_dense_retrieval_summary.json"))
    parser.add_argument("--index", type=Path, default=Path("results/day1_dense.faiss"))
    parser.add_argument("--embeddings", type=Path, default=Path("results/day1_dense_embeddings.npy"))
    parser.add_argument("--manifest", type=Path, default=Path("results/day1_dense_manifest.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this server experiment")

    chunks = read_jsonl(args.chunks)
    questions = read_jsonl(args.questions)

    model_load_started = time.perf_counter()
    retriever = DenseRetriever(
        chunks,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        device="cuda",
        batch_size=args.batch_size,
    )
    model_load_seconds = time.perf_counter() - model_load_started

    index_build_started = time.perf_counter()
    retriever.build()
    index_build_seconds = time.perf_counter() - index_build_started
    retriever.save_artifacts(
        index_path=args.index,
        embeddings_path=args.embeddings,
        manifest_path=args.manifest,
    )

    assert retriever.document_embeddings is not None
    vector_norms = np.linalg.norm(retriever.document_embeddings, axis=1)
    np.testing.assert_allclose(vector_norms, 1.0, atol=1e-5)

    traces: list[dict[str, Any]] = []
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

    write_jsonl(args.output, traces)
    question_count = len(traces)
    document_ranks = [trace["gold_document_best_rank"] for trace in traces]
    answer_ranks = [trace["gold_answer_best_rank"] for trace in traces]
    summary = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "chunks_sha256": hashlib.sha256(args.chunks.read_bytes()).hexdigest(),
        "chunk_count": len(chunks),
        "embedding_dimension": int(retriever.document_embeddings.shape[1]),
        "document_vector_norm": {
            "minimum": round(float(vector_norms.min()), 6),
            "maximum": round(float(vector_norms.max()), 6),
        },
        "index": {
            "type": "IndexFlatIP",
            "ntotal": int(retriever.index.ntotal) if retriever.index is not None else 0,
            "top_k": args.top_k,
        },
        "metrics": {
            "question_count": question_count,
            "document_recall_at_1": sum(rank == 1 for rank in document_ranks) / question_count,
            "document_recall_at_3": sum(rank is not None and rank <= 3 for rank in document_ranks) / question_count,
            "answer_recall_at_1": sum(rank == 1 for rank in answer_ranks) / question_count,
            "answer_recall_at_3": sum(rank is not None and rank <= 3 for rank in answer_ranks) / question_count,
            "document_mrr": round(sum(reciprocal_rank(rank) for rank in document_ranks) / question_count, 6),
            "answer_mrr": round(sum(reciprocal_rank(rank) for rank in answer_ranks) / question_count, 6),
        },
        "per_question": [
            {
                "question_id": trace["question_id"],
                "gold_document_best_rank": trace["gold_document_best_rank"],
                "gold_answer_best_rank": trace["gold_answer_best_rank"],
                "top1_chunk_id": trace["retrieved_ids"][0],
                "top1_score": trace["retrieval_scores"][0],
            }
            for trace in traces
        ],
        "timing": {
            "model_load_seconds": round(model_load_seconds, 3),
            "document_embedding_and_index_seconds": round(index_build_seconds, 3),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "faiss": faiss.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "artifacts": {
            "index": str(args.index),
            "embeddings": str(args.embeddings),
            "manifest": str(args.manifest),
            "traces": str(args.output),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
