"""Run corrected watermark query triplets across four retrieval pipelines."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever
from qwen_reranker import DEFAULT_INSTRUCTION, QwenReranker
from rrf_fusion import reciprocal_rank_fusion
from watermark_retrieval_metrics import (
    DEFAULT_KS,
    summarize_retriever,
    target_diagnostics,
    transfer_matrix,
)


EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_REVISION = "5340c0261aa49a842d1bff01db91ce407bda87a2"
RETRIEVERS = ("bm25", "dense", "rrf_hybrid", "qwen3_reranker")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
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


def query_cases(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for pair in pairs:
        for condition, field in (
            ("normal", "normal_query"),
            ("trigger_only", "trigger_only_query"),
            ("verification", "verification_query"),
        ):
            cases.append(
                {
                    **pair,
                    "condition": condition,
                    "query": pair[field],
                }
            )
    return cases


def attach_diagnostics(
    *,
    retriever: str,
    case: dict[str, Any],
    trace: dict[str, Any],
    watermark_chunk_ids: set[str],
) -> dict[str, Any]:
    diagnostics = target_diagnostics(
        trace["results"],
        target_chunk_id=case["target_chunk_id"],
        watermark_chunk_ids=watermark_chunk_ids,
    )
    return {
        "retriever": retriever,
        "pair_id": case["pair_id"],
        "category": case["category"],
        "condition": case["condition"],
        "trigger": case["trigger"],
        "query": case["query"],
        "normal_query": case["normal_query"],
        "trigger_only_query": case["trigger_only_query"],
        "verification_query": case["verification_query"],
        "target_chunk_id": case["target_chunk_id"],
        "target_document_id": case["target_document_id"],
        "clean_gold_chunk_ids": case["clean_gold_chunk_ids"],
        "source_policy_topic": case["source_policy_topic"],
        "source_fact": case["source_fact"],
        "verification_code": case["verification_code"],
        "target_fact": case["target_fact"],
        **diagnostics,
        "trace": trace,
    }


def comparison_csv_rows(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (trace["retriever"], trace["pair_id"], trace["condition"]): trace
        for trace in traces
    }
    pair_ids = sorted({trace["pair_id"] for trace in traces})
    rows = []
    for retriever in RETRIEVERS:
        for pair_id in pair_ids:
            normal = by_key[(retriever, pair_id, "normal")]
            trigger_only = by_key[(retriever, pair_id, "trigger_only")]
            verification = by_key[(retriever, pair_id, "verification")]
            row: dict[str, Any] = {
                "retriever": retriever,
                "pair_id": pair_id,
                "category": normal["category"],
                "trigger": normal["trigger"],
                "target_chunk_id": normal["target_chunk_id"],
                "clean_gold_chunk_ids": "|".join(normal["clean_gold_chunk_ids"]),
                "normal_target_rank": normal["target_rank"],
                "trigger_only_target_rank": trigger_only["target_rank"],
                "verification_target_rank": verification["target_rank"],
                "trigger_only_rank_gain": (
                    normal["target_rank"] - trigger_only["target_rank"]
                ),
                "verification_rank_gain": (
                    normal["target_rank"] - verification["target_rank"]
                ),
                "normal_target_score": normal["target_score"],
                "trigger_only_target_score": trigger_only["target_score"],
                "verification_target_score": verification["target_score"],
                "trigger_only_target_gap_to_next": trigger_only[
                    "target_gap_to_next"
                ],
                "verification_target_gap_to_next": verification[
                    "target_gap_to_next"
                ],
            }
            for k in DEFAULT_KS:
                row[f"normal_exact_target_hit_at_{k}"] = normal[f"target_hit_at_{k}"]
                row[f"normal_any_watermark_hit_at_{k}"] = normal[
                    f"any_watermark_hit_at_{k}"
                ]
                row[f"trigger_only_target_hit_at_{k}"] = trigger_only[
                    f"target_hit_at_{k}"
                ]
                row[f"verification_target_hit_at_{k}"] = verification[
                    f"target_hit_at_{k}"
                ]
            rows.append(row)
    return rows


def rank_change_summary(
    traces: list[dict[str, Any]],
    *,
    condition: str,
) -> dict[str, Any]:
    keyed = {
        (trace["retriever"], trace["pair_id"], trace["condition"]): trace
        for trace in traces
    }
    pair_ids = sorted(
        {
            trace["pair_id"]
            for trace in traces
            if trace["condition"] == condition
        }
    )
    changes = [
        keyed[("rrf_hybrid", pair_id, condition)]["target_rank"]
        - keyed[("qwen3_reranker", pair_id, condition)]["target_rank"]
        for pair_id in pair_ids
    ]
    return {
        "condition": condition,
        "positive_means_reranker_improved_rank": True,
        "mean_rank_change": round(mean(changes), 6),
        "improved_count": sum(change > 0 for change in changes),
        "unchanged_count": sum(change == 0 for change in changes),
        "degraded_count": sum(change < 0 for change in changes),
        "per_pair": dict(zip(pair_ids, changes)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/watermarked/day2_retrieval_chunks.jsonl"),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/eval/day2_watermark_query_triplets.jsonl"),
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("results/day2_watermark_retrieval_traces.jsonl"),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("results/day2_watermark_retrieval_comparison.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/day2_watermark_retrieval_summary.json"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker-max-length", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import faiss
    import sentence_transformers
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Dense retrieval and Qwen3 Reranker")

    chunks = read_jsonl(args.chunks)
    pairs = read_jsonl(args.pairs)
    cases = query_cases(pairs)
    if len(pairs) < 20:
        raise ValueError("The experiment requires at least 20 query triplets")
    if len(cases) != len(pairs) * 3:
        raise AssertionError("Each sample must generate exactly three query conditions")

    watermark_chunk_ids = {
        chunk["chunk_id"]
        for chunk in chunks
        if chunk.get("metadata", {}).get("is_watermarked")
    }
    expected_targets = {pair["target_chunk_id"] for pair in pairs}
    if watermark_chunk_ids != expected_targets:
        raise ValueError("Watermark Chunk IDs and pair target IDs do not match")

    experiment_started = time.perf_counter()
    bm25_build_started = time.perf_counter()
    bm25 = BM25Retriever(chunks)
    bm25_build_seconds = time.perf_counter() - bm25_build_started

    dense_load_started = time.perf_counter()
    dense = DenseRetriever(
        chunks,
        model_id=EMBEDDING_MODEL_ID,
        revision=EMBEDDING_REVISION,
        device="cuda",
        batch_size=args.batch_size,
    )
    dense_model_load_seconds = time.perf_counter() - dense_load_started
    dense_build_started = time.perf_counter()
    dense.build()
    dense_build_seconds = time.perf_counter() - dense_build_started

    all_traces: list[dict[str, Any]] = []
    hybrid_by_case: dict[tuple[str, str], dict[str, Any]] = {}
    for case in cases:
        bm25_trace = bm25.search(case["query"], top_k=len(chunks))
        dense_trace = dense.search(case["query"], top_k=len(chunks))
        hybrid_trace = reciprocal_rank_fusion(
            {
                "bm25": bm25_trace["results"],
                "dense": dense_trace["results"],
            },
            rrf_k=args.rrf_k,
            top_k=len(chunks),
        )
        case_key = (case["pair_id"], case["condition"])
        hybrid_by_case[case_key] = hybrid_trace
        all_traces.extend(
            [
                attach_diagnostics(
                    retriever="bm25",
                    case=case,
                    trace=bm25_trace,
                    watermark_chunk_ids=watermark_chunk_ids,
                ),
                attach_diagnostics(
                    retriever="dense",
                    case=case,
                    trace=dense_trace,
                    watermark_chunk_ids=watermark_chunk_ids,
                ),
                attach_diagnostics(
                    retriever="rrf_hybrid",
                    case=case,
                    trace=hybrid_trace,
                    watermark_chunk_ids=watermark_chunk_ids,
                ),
            ]
        )

    del dense
    gc.collect()
    torch.cuda.empty_cache()

    reranker = QwenReranker(
        model_id=RERANKER_MODEL_ID,
        revision=RERANKER_REVISION,
        instruction=DEFAULT_INSTRUCTION,
        max_length=args.reranker_max_length,
    )
    reranker_started = time.perf_counter()
    for case in cases:
        case_key = (case["pair_id"], case["condition"])
        reranker_trace = reranker.rerank(
            case["query"],
            hybrid_by_case[case_key]["results"],
        )
        all_traces.append(
            attach_diagnostics(
                retriever="qwen3_reranker",
                case=case,
                trace=reranker_trace,
                watermark_chunk_ids=watermark_chunk_ids,
            )
        )
    reranker_scoring_seconds = time.perf_counter() - reranker_started

    all_traces.sort(
        key=lambda trace: (
            RETRIEVERS.index(trace["retriever"]),
            trace["pair_id"],
            trace["condition"],
        )
    )
    write_jsonl(args.traces, all_traces)
    comparison_rows = comparison_csv_rows(all_traces)
    write_csv(args.comparison, comparison_rows)

    per_retriever = {
        retriever: summarize_retriever(
            [trace for trace in all_traces if trace["retriever"] == retriever]
        )
        for retriever in RETRIEVERS
    }
    trigger_only_traces = [
        trace for trace in all_traces if trace["condition"] == "trigger_only"
    ]
    verification_traces = [
        trace for trace in all_traces if trace["condition"] == "verification"
    ]
    summary = {
        "experiment": "Day 2 corrected watermark retrieval geometry",
        "watermark_type": (
            "reranker-aware semantic Canary plus trigger-only control"
        ),
        "metric_definition": {
            "target_hit_at_k": "exact target watermark Chunk rank <= k",
            "normal_exact_target_false_trigger_at_k": (
                "normal business query retrieves its exact Canary in Top-k"
            ),
            "normal_any_watermark_exposure_at_k": (
                "normal business query retrieves any Canary in Top-k"
            ),
            "target_gap_to_next": "target score minus the score at the next rank",
            "transfer": "P(target Retriever hit | source Retriever hit)",
            "trigger_only_rank_gain": (
                "normal target rank minus same-business-query-with-trigger rank"
            ),
            "verification_rank_gain": (
                "normal target rank minus dedicated verification-query rank"
            ),
        },
        "data": {
            "chunk_count": len(chunks),
            "clean_chunk_count": len(chunks) - len(watermark_chunk_ids),
            "watermark_chunk_count": len(watermark_chunk_ids),
            "query_triplet_count": len(pairs),
            "query_count": len(cases),
            "chunks_sha256": hashlib.sha256(args.chunks.read_bytes()).hexdigest(),
            "pairs_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        },
        "models": {
            "embedding": {
                "id": EMBEDDING_MODEL_ID,
                "revision": EMBEDDING_REVISION,
                "normalization": True,
                "index": "FAISS IndexFlatIP",
            },
            "reranker": {
                "id": RERANKER_MODEL_ID,
                "revision": RERANKER_REVISION,
                "instruction": DEFAULT_INSTRUCTION,
                "score": "yes_logit - no_logit",
                "candidate_depth": len(chunks),
            },
            "bm25": {
                "k1": bm25.k1,
                "b": bm25.b,
                "tokenizer": "NFKC lowercase; Chinese unigrams+bigrams; alphanumeric spans",
            },
            "rrf": {
                "k": args.rrf_k,
                "source_depth": len(chunks),
            },
        },
        "per_retriever": per_retriever,
        "cross_retriever_transfer": {
            "trigger_only": {
                f"at_{k}": transfer_matrix(trigger_only_traces, k=k)
                for k in DEFAULT_KS
            },
            "verification": {
                f"at_{k}": transfer_matrix(verification_traces, k=k)
                for k in DEFAULT_KS
            },
        },
        "reranker_vs_hybrid": {
            "normal": rank_change_summary(all_traces, condition="normal"),
            "trigger_only": rank_change_summary(
                all_traces,
                condition="trigger_only",
            ),
            "verification": rank_change_summary(
                all_traces,
                condition="verification",
            ),
        },
        "timing_seconds": {
            "bm25_build": round(bm25_build_seconds, 6),
            "dense_model_load": round(dense_model_load_seconds, 3),
            "dense_document_embedding_and_index": round(dense_build_seconds, 3),
            "reranker_model_load": round(reranker.model_load_seconds, 3),
            "reranker_60x_full_corpus_scoring": round(
                reranker_scoring_seconds,
                3,
            ),
            "total": round(time.perf_counter() - experiment_started, 3),
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
            "traces": str(args.traces),
            "comparison": str(args.comparison),
            "summary": str(args.summary),
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
