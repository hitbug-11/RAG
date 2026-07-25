"""Run a paired Chunk Size × Overlap watermark boundary-stress experiment.

Twenty long synthetic carrier documents contain the same corrected semantic
Canaries used by the Day 2 retrieval experiment.  The Canary starts are fixed
per pair, while nine character-based splitter configurations vary max Chunk
size (256/512/1024) and overlap (0/64/128).  The experiment records whether the
trigger-to-code evidence span survives in any single Chunk before evaluating
BM25, Dense, RRF, and Qwen3 Reranker retrieval.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import platform
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from build_chunks import BREAK_CHARACTERS, split_document


EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_REVISION = "5340c0261aa49a842d1bff01db91ce407bda87a2"

CHUNK_SIZES = (256, 512, 1024)
OVERLAPS = (0, 64, 128)
RETRIEVERS = ("bm25", "dense", "rrf_hybrid", "qwen3_reranker")
CONDITIONS = ("normal", "trigger_only", "verification")
KS = (1, 5, 10, 20)
RERANKER_CANDIDATE_DEPTH = 20
DOCUMENT_LENGTH = 1800
MARKER_STARTS = tuple(180 + 72 * index for index in range(20))

ABLATION_FIELDS = (
    "ablation",
    "position",
    "chunk_size_chars",
    "overlap_chars",
    "corpus_chunk_count",
    "retriever",
    "pair_id",
    "category",
    "condition",
    "trigger",
    "target_chunk_id",
    "joint_evidence_chunk_ids",
    "joint_evidence_preserved",
    "joint_evidence_copy_count",
    "trigger_chunk_ids",
    "code_chunk_ids",
    "clean_gold_chunk_ids",
    "target_rank",
    "target_score",
    "target_gap_to_next",
    "target_gap_to_top1",
    "trigger_fragment_rank",
    "code_fragment_rank",
    "target_hit_at_1",
    "target_hit_at_5",
    "target_hit_at_10",
    "target_hit_at_20",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_context = (
        gzip.open(path, "wt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open("w", encoding="utf-8")
    )
    with handle_context as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=names,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def watermark_block(pair: dict[str, Any]) -> str:
    """Keep trigger and code in one long, punctuation-safe evidence span."""
    return (
        f"版权核验记录：版本追踪标记“{pair['trigger']}”"
        "对应一条仅用于知识库授权副本识别的内部记录，"
        "该记录不描述任何可用于外部问答的事实，"
        "也不得用于回答与授权核验无关的问题，"
        "为避免核验信息与普通正文混淆，"
        "中间保留一段无实际含义的审计说明，"
        "审计说明只确认标记与口令属于同一条授权记录，"
        f"该标记最终对应的核验口令为“{pair['verification_code']}”，"
    )


def build_carrier_documents(
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if len(pairs) != len(MARKER_STARTS):
        raise ValueError(
            f"Expected {len(MARKER_STARTS)} pairs, got {len(pairs)}"
        )
    documents = []
    spans: dict[str, dict[str, Any]] = {}
    for pair, marker_start in zip(pairs, MARKER_STARTS):
        block = watermark_block(pair)
        unit = (
            f"合成审计载荷{pair['pair_id']}，"
            "本段仅用于确定性切分边界测试，无实际规则含义，"
        )
        carrier = (unit * (DOCUMENT_LENGTH // len(unit) + 2))[:DOCUMENT_LENGTH]
        text = (
            carrier[:marker_start]
            + block
            + carrier[marker_start + len(block) :]
        )
        if len(text) != DOCUMENT_LENGTH:
            raise AssertionError("Carrier replacement changed document length")
        if any(character in text for character in BREAK_CHARACTERS):
            raise AssertionError("Carrier must not contain preferred break characters")
        if pair["source_fact"] in text:
            raise AssertionError(f"Canary copied business answer: {pair['pair_id']}")

        trigger_start = marker_start + block.index(pair["trigger"])
        trigger_end = trigger_start + len(pair["trigger"])
        code_start = marker_start + block.index(pair["verification_code"])
        code_end = code_start + len(pair["verification_code"])
        if not trigger_end < code_start:
            raise AssertionError("Trigger must precede verification code")

        document_id = f"chunk-ablation-{pair['pair_id']}"
        documents.append(
            {
                "document_id": document_id,
                "title": f"边界压力载荷 {pair['pair_id']}",
                "source": f"synthetic/chunk-ablation/{pair['pair_id']}",
                "version": "2026-07",
                "text": text,
                "metadata": {
                    "is_watermarked": True,
                    "pair_id": pair["pair_id"],
                },
            }
        )
        spans[pair["pair_id"]] = {
            "document_id": document_id,
            "marker_start": marker_start,
            "marker_end": marker_start + len(block),
            "marker_length": len(block),
            "trigger_start": trigger_start,
            "trigger_end": trigger_end,
            "code_start": code_start,
            "code_end": code_end,
            "joint_start": trigger_start,
            "joint_end": code_end,
            "joint_length": code_end - trigger_start,
        }
    return documents, spans


def chunk_configuration(
    clean_documents: list[dict[str, Any]],
    carrier_documents: list[dict[str, Any]],
    spans: dict[str, dict[str, Any]],
    *,
    chunk_size: int,
    overlap: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    min_chars = chunk_size * 3 // 4
    if not 0 <= overlap < min_chars <= chunk_size:
        raise ValueError("Require 0 <= overlap < min_chars <= chunk_size")

    documents = [*clean_documents, *carrier_documents]
    chunks = [
        asdict(chunk)
        for document in documents
        for chunk in split_document(
            document,
            max_chars=chunk_size,
            overlap_chars=overlap,
            min_chars=min_chars,
        )
    ]
    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise AssertionError("Chunk IDs must be unique within a configuration")

    targets: dict[str, dict[str, Any]] = {}
    for pair_id, span in spans.items():
        document_chunks = [
            chunk
            for chunk in chunks
            if chunk["document_id"] == span["document_id"]
        ]

        def covering(start: int, end: int) -> list[str]:
            return [
                chunk["chunk_id"]
                for chunk in document_chunks
                if chunk["start_char"] <= start and chunk["end_char"] >= end
            ]

        trigger_ids = covering(span["trigger_start"], span["trigger_end"])
        code_ids = covering(span["code_start"], span["code_end"])
        joint_ids = covering(span["joint_start"], span["joint_end"])
        targets[pair_id] = {
            "trigger_chunk_ids": trigger_ids,
            "code_chunk_ids": code_ids,
            "joint_evidence_chunk_ids": joint_ids,
            "joint_evidence_preserved": bool(joint_ids),
            "joint_evidence_copy_count": len(joint_ids),
        }

    for chunk in chunks:
        chunk["metadata"] = {
            **chunk["metadata"],
            "chunk_size_chars": chunk_size,
            "overlap_chars": overlap,
        }
    return chunks, targets


def query_cases(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for pair in pairs:
        for condition, field in (
            ("normal", "normal_query"),
            ("trigger_only", "trigger_only_query"),
            ("verification", "verification_query"),
        ):
            cases.append({**pair, "condition": condition, "query": pair[field]})
    return cases


def best_rank(results: list[dict[str, Any]], chunk_ids: Iterable[str]) -> int | None:
    targets = set(chunk_ids)
    return next(
        (
            index
            for index, result in enumerate(results, start=1)
            if result["chunk_id"] in targets
        ),
        None,
    )


def target_set_diagnostics(
    results: list[dict[str, Any]],
    *,
    target_chunk_ids: Iterable[str],
) -> dict[str, Any]:
    targets = set(target_chunk_ids)
    target_index = next(
        (
            index
            for index, result in enumerate(results)
            if result["chunk_id"] in targets
        ),
        None,
    )
    rank = None if target_index is None else target_index + 1
    score = None if target_index is None else float(results[target_index]["score"])
    next_score = (
        None
        if target_index is None or target_index + 1 >= len(results)
        else float(results[target_index + 1]["score"])
    )
    top_score = None if not results else float(results[0]["score"])
    row: dict[str, Any] = {
        "target_rank": rank,
        "target_score": score,
        "target_gap_to_next": (
            None
            if score is None or next_score is None
            else round(score - next_score, 9)
        ),
        "target_gap_to_top1": (
            None
            if score is None or top_score is None
            else round(score - top_score, 9)
        ),
    }
    for k in KS:
        row[f"target_hit_at_{k}"] = rank is not None and rank <= k
    return row


def compact_retrieval_trace(trace: dict[str, Any]) -> dict[str, Any]:
    result_fields = {
        "rank",
        "faiss_id",
        "chunk_id",
        "score",
        "document_length",
        "matched_term_count",
        "source_count",
        "best_source_rank",
        "source_ranks",
        "source_contributions",
        "hybrid_rank",
        "hybrid_score",
        "reranker_logit_difference",
        "relevance_probability",
        "input_tokens",
    }
    omitted = {
        "formatted_pairs",
        "model_prefix",
        "model_suffix",
        "retrieved_ids",
        "retrieval_scores",
        "results",
    }
    return {
        **{key: value for key, value in trace.items() if key not in omitted},
        "result_depth_before_compaction": len(trace["results"]),
        "results": [
            {
                key: value
                for key, value in result.items()
                if key in result_fields
            }
            for result in trace["results"][:RERANKER_CANDIDATE_DEPTH]
        ],
    }


def make_trace(
    *,
    chunk_size: int,
    overlap: int,
    corpus_size: int,
    retriever: str,
    case: dict[str, Any],
    target: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    results = trace["results"]
    return {
        "ablation": "chunk_size_overlap",
        "chunk_size_chars": chunk_size,
        "overlap_chars": overlap,
        "corpus_chunk_count": corpus_size,
        "retriever": retriever,
        "pair_id": case["pair_id"],
        "category": case["category"],
        "condition": case["condition"],
        "trigger": case["trigger"],
        "query": case["query"],
        "clean_gold_chunk_ids": case["clean_gold_chunk_ids"],
        **target,
        **target_set_diagnostics(
            results,
            target_chunk_ids=target["joint_evidence_chunk_ids"],
        ),
        "trigger_fragment_rank": best_rank(
            results,
            target["trigger_chunk_ids"],
        ),
        "code_fragment_rank": best_rank(
            results,
            target["code_chunk_ids"],
        ),
        "trace": compact_retrieval_trace(trace),
    }


def ablation_rows(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for trace in traces:
        rows.append(
            {
                "ablation": trace["ablation"],
                "position": "",
                "chunk_size_chars": trace["chunk_size_chars"],
                "overlap_chars": trace["overlap_chars"],
                "corpus_chunk_count": trace["corpus_chunk_count"],
                "retriever": trace["retriever"],
                "pair_id": trace["pair_id"],
                "category": trace["category"],
                "condition": trace["condition"],
                "trigger": trace["trigger"],
                "target_chunk_id": "",
                "joint_evidence_chunk_ids": "|".join(
                    trace["joint_evidence_chunk_ids"]
                ),
                "joint_evidence_preserved": trace["joint_evidence_preserved"],
                "joint_evidence_copy_count": trace["joint_evidence_copy_count"],
                "trigger_chunk_ids": "|".join(trace["trigger_chunk_ids"]),
                "code_chunk_ids": "|".join(trace["code_chunk_ids"]),
                "clean_gold_chunk_ids": "|".join(trace["clean_gold_chunk_ids"]),
                "target_rank": trace["target_rank"],
                "target_score": trace["target_score"],
                "target_gap_to_next": trace["target_gap_to_next"],
                "target_gap_to_top1": trace["target_gap_to_top1"],
                "trigger_fragment_rank": trace["trigger_fragment_rank"],
                "code_fragment_rank": trace["code_fragment_rank"],
                **{
                    f"target_hit_at_{k}": trace[f"target_hit_at_{k}"]
                    for k in KS
                },
            }
        )
    return rows


def merge_ablation_rows(
    existing_path: Path,
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing: list[dict[str, Any]] = []
    if existing_path.exists():
        with existing_path.open("r", encoding="utf-8", newline="") as handle:
            existing = [
                row
                for row in csv.DictReader(handle)
                if row.get("ablation") != "chunk_size_overlap"
            ]
    normalized = [
        {field: row.get(field, "") for field in ABLATION_FIELDS}
        for row in [*existing, *new_rows]
    ]
    return normalized


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def summarize_groups(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            for retriever in RETRIEVERS:
                for condition in CONDITIONS:
                    group = [
                        trace
                        for trace in traces
                        if trace["chunk_size_chars"] == chunk_size
                        and trace["overlap_chars"] == overlap
                        and trace["retriever"] == retriever
                        and trace["condition"] == condition
                    ]
                    if not group:
                        raise ValueError(
                            f"Missing group {chunk_size}/{overlap}/"
                            f"{retriever}/{condition}"
                        )
                    preserved = [
                        trace
                        for trace in group
                        if trace["joint_evidence_preserved"]
                    ]
                    preserved_ranks = [
                        trace["target_rank"]
                        for trace in preserved
                        if trace["target_rank"] is not None
                    ]
                    row: dict[str, Any] = {
                        "chunk_size_chars": chunk_size,
                        "overlap_chars": overlap,
                        "corpus_chunk_count": group[0]["corpus_chunk_count"],
                        "retriever": retriever,
                        "condition": condition,
                        "sample_count": len(group),
                        "joint_evidence_preservation_rate": round(
                            mean(
                                bool(trace["joint_evidence_preserved"])
                                for trace in group
                            ),
                            6,
                        ),
                        "mean_joint_evidence_copy_count": round(
                            mean(trace["joint_evidence_copy_count"] for trace in group),
                            6,
                        ),
                        "target_mrr_all_pairs": round(
                            mean(
                                reciprocal_rank(trace["target_rank"])
                                for trace in group
                            ),
                            6,
                        ),
                        "target_mrr_preserved_only": (
                            None
                            if not preserved
                            else round(
                                mean(
                                    reciprocal_rank(trace["target_rank"])
                                    for trace in preserved
                                ),
                                6,
                            )
                        ),
                        "median_target_rank_preserved_only": (
                            None
                            if not preserved_ranks
                            else median(preserved_ranks)
                        ),
                    }
                    for k in KS:
                        row[f"target_hit_at_{k}_all_pairs"] = round(
                            mean(bool(trace[f"target_hit_at_{k}"]) for trace in group),
                            6,
                        )
                        row[f"target_hit_at_{k}_preserved_only"] = (
                            None
                            if not preserved
                            else round(
                                mean(
                                    bool(trace[f"target_hit_at_{k}"])
                                    for trace in preserved
                                ),
                                6,
                            )
                        )
                    rows.append(row)
    return rows


def pca_coordinates(matrix: np.ndarray) -> tuple[np.ndarray, list[float]]:
    if matrix.ndim != 2 or len(matrix) < 2:
        raise ValueError("PCA requires at least two row vectors")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, components = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    coordinates = centered @ components[:2].T
    variances = singular_values**2
    explained = variances[:2] / variances.sum()
    return coordinates.astype("float32"), [
        round(float(value), 8) for value in explained
    ]


def select_best_vector(
    *,
    candidate_ids: list[str],
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
) -> tuple[str, np.ndarray, float] | None:
    if not candidate_ids:
        return None
    index_by_id = {
        chunk["chunk_id"]: index for index, chunk in enumerate(chunks)
    }
    candidates = [
        (
            chunk_id,
            embeddings[index_by_id[chunk_id]],
            float(embeddings[index_by_id[chunk_id]] @ query_embedding),
        )
        for chunk_id in candidate_ids
    ]
    return max(candidates, key=lambda item: (item[2], item[0]))


def build_pca_records(
    vector_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[float]]:
    matrix = np.stack([record.pop("_vector") for record in vector_records])
    coordinates, explained = pca_coordinates(matrix)
    rows = []
    for record, (pc1, pc2) in zip(vector_records, coordinates):
        rows.append(
            {
                **record,
                "pc1": round(float(pc1), 8),
                "pc2": round(float(pc2), 8),
            }
        )
    return rows, explained


def displacement_summary(
    pca_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    queries = {
        row["pair_id"]: row
        for row in pca_rows
        if row["role"] == "verification_query"
    }
    rows = []
    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            evidence = [
                row
                for row in pca_rows
                if row["role"] != "verification_query"
                and row["chunk_size_chars"] == chunk_size
                and row["overlap_chars"] == overlap
            ]
            joint = [row for row in evidence if row["role"] == "joint_evidence"]
            fragments = [row for row in evidence if "fragment" in row["role"]]
            rows.append(
                {
                    "chunk_size_chars": chunk_size,
                    "overlap_chars": overlap,
                    "joint_evidence_count": len(joint),
                    "fragment_point_count": len(fragments),
                    "mean_joint_cosine_to_query": (
                        None
                        if not joint
                        else round(mean(row["cosine_to_query"] for row in joint), 6)
                    ),
                    "mean_fragment_cosine_to_query": (
                        None
                        if not fragments
                        else round(
                            mean(row["cosine_to_query"] for row in fragments),
                            6,
                        )
                    ),
                    "mean_joint_pca_distance_to_query": (
                        None
                        if not joint
                        else round(
                            mean(
                                (
                                    (row["pc1"] - queries[row["pair_id"]]["pc1"]) ** 2
                                    + (
                                        row["pc2"]
                                        - queries[row["pair_id"]]["pc2"]
                                    )
                                    ** 2
                                )
                                ** 0.5
                                for row in joint
                            ),
                            6,
                        )
                    ),
                }
            )
    return rows


def plot_pca(
    pca_rows: list[dict[str, Any]],
    *,
    explained: list[float],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    queries = [row for row in pca_rows if row["role"] == "verification_query"]
    colors = {0: "#4C78A8", 64: "#F58518", 128: "#54A24B"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for axis, chunk_size in zip(axes, CHUNK_SIZES):
        axis.scatter(
            [row["pc1"] for row in queries],
            [row["pc2"] for row in queries],
            facecolors="none",
            edgecolors="#4D4D4D",
            s=28,
            linewidths=0.8,
            label="Verification query",
        )
        for overlap in OVERLAPS:
            evidence = [
                row
                for row in pca_rows
                if row["role"] != "verification_query"
                and row["chunk_size_chars"] == chunk_size
                and row["overlap_chars"] == overlap
            ]
            joint = [row for row in evidence if row["role"] == "joint_evidence"]
            fragments = [row for row in evidence if "fragment" in row["role"]]
            axis.scatter(
                [row["pc1"] for row in joint],
                [row["pc2"] for row in joint],
                color=colors[overlap],
                marker="o",
                s=24,
                alpha=0.75,
                label=f"Overlap {overlap}: joint",
            )
            axis.scatter(
                [row["pc1"] for row in fragments],
                [row["pc2"] for row in fragments],
                color=colors[overlap],
                marker="x",
                s=26,
                linewidths=0.9,
                alpha=0.8,
                label=f"Overlap {overlap}: fragment",
            )
        axis.set_title(f"Chunk size {chunk_size} chars")
        axis.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        axis.grid(alpha=0.18, linewidth=0.6)
    axes[0].set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "PCA: verification queries, intact evidence, and boundary fragments",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-documents",
        type=Path,
        default=Path("data/clean/day1_knowledge_base.jsonl"),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/eval/day2_watermark_query_triplets.jsonl"),
    )
    parser.add_argument(
        "--carrier-output",
        type=Path,
        default=Path(
            "data/watermarked/day2_chunking_ablation_documents.jsonl"
        ),
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("results/day2_chunking_ablation_traces.jsonl.gz"),
    )
    parser.add_argument(
        "--ablation-csv",
        type=Path,
        default=Path("results/retrieval_ablation.csv"),
    )
    parser.add_argument(
        "--group-summary-csv",
        type=Path,
        default=Path("results/day2_chunking_ablation_group_summary.csv"),
    )
    parser.add_argument(
        "--pca-coordinates",
        type=Path,
        default=Path("results/day2_embedding_pca_coordinates.csv"),
    )
    parser.add_argument(
        "--displacement-summary",
        type=Path,
        default=Path("results/day2_embedding_displacement_summary.csv"),
    )
    parser.add_argument(
        "--visualization",
        type=Path,
        default=Path("results/embedding_visualization.png"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/day2_chunking_ablation_summary.json"),
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

    from bm25_retriever import BM25Retriever
    from dense_retriever import DenseRetriever
    from qwen_reranker import DEFAULT_INSTRUCTION, QwenReranker
    from rrf_fusion import reciprocal_rank_fusion

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Dense retrieval and Qwen3 Reranker")

    clean_documents = read_jsonl(args.clean_documents)
    pairs = read_jsonl(args.pairs)
    cases = query_cases(pairs)
    carrier_documents, source_spans = build_carrier_documents(pairs)
    write_jsonl(args.carrier_output, carrier_documents)

    started = time.perf_counter()
    traces: list[dict[str, Any]] = []
    hybrid_candidates: dict[
        tuple[int, int, str, str], tuple[dict[str, Any], dict[str, Any], int]
    ] = {}
    vector_records: list[dict[str, Any]] = []
    query_embeddings_by_pair: dict[str, np.ndarray] = {}
    configuration_metadata: dict[str, Any] = {}

    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            configuration_started = time.perf_counter()
            chunks, targets = chunk_configuration(
                clean_documents,
                carrier_documents,
                source_spans,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            bm25 = BM25Retriever(chunks)
            dense = DenseRetriever(
                chunks,
                model_id=EMBEDDING_MODEL_ID,
                revision=EMBEDDING_REVISION,
                device="cuda",
                batch_size=args.batch_size,
            )
            dense.build()
            if dense.document_embeddings is None:
                raise AssertionError("Dense document embeddings are missing")

            verification_queries = [pair["verification_query"] for pair in pairs]
            verification_embeddings = dense.model.encode(
                verification_queries,
                prompt_name="query",
                batch_size=args.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            if not query_embeddings_by_pair:
                for pair, vector in zip(pairs, verification_embeddings):
                    query_embeddings_by_pair[pair["pair_id"]] = vector
                    vector_records.append(
                        {
                            "role": "verification_query",
                            "pair_id": pair["pair_id"],
                            "chunk_size_chars": "",
                            "overlap_chars": "",
                            "chunk_id": "",
                            "joint_evidence_preserved": "",
                            "cosine_to_query": 1.0,
                            "_vector": vector,
                        }
                    )

            for pair, query_vector in zip(pairs, verification_embeddings):
                target = targets[pair["pair_id"]]
                if target["joint_evidence_chunk_ids"]:
                    selected = select_best_vector(
                        candidate_ids=target["joint_evidence_chunk_ids"],
                        chunks=chunks,
                        embeddings=dense.document_embeddings,
                        query_embedding=query_vector,
                    )
                    selections = [("joint_evidence", selected)]
                else:
                    selections = [
                        (
                            "trigger_fragment",
                            select_best_vector(
                                candidate_ids=target["trigger_chunk_ids"],
                                chunks=chunks,
                                embeddings=dense.document_embeddings,
                                query_embedding=query_vector,
                            ),
                        ),
                        (
                            "code_fragment",
                            select_best_vector(
                                candidate_ids=target["code_chunk_ids"],
                                chunks=chunks,
                                embeddings=dense.document_embeddings,
                                query_embedding=query_vector,
                            ),
                        ),
                    ]
                seen_selected_ids: set[str] = set()
                for role, selected in selections:
                    if selected is None:
                        continue
                    chunk_id, vector, cosine = selected
                    if chunk_id in seen_selected_ids:
                        continue
                    seen_selected_ids.add(chunk_id)
                    vector_records.append(
                        {
                            "role": role,
                            "pair_id": pair["pair_id"],
                            "chunk_size_chars": chunk_size,
                            "overlap_chars": overlap,
                            "chunk_id": chunk_id,
                            "joint_evidence_preserved": target[
                                "joint_evidence_preserved"
                            ],
                            "cosine_to_query": round(cosine, 8),
                            "_vector": vector,
                        }
                    )

            for case in cases:
                target = targets[case["pair_id"]]
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
                key = (
                    chunk_size,
                    overlap,
                    case["pair_id"],
                    case["condition"],
                )
                hybrid_candidates[key] = (
                    {
                        **hybrid_trace,
                        "results": hybrid_trace["results"][
                            :RERANKER_CANDIDATE_DEPTH
                        ],
                    },
                    target,
                    len(chunks),
                )
                traces.extend(
                    [
                        make_trace(
                            chunk_size=chunk_size,
                            overlap=overlap,
                            corpus_size=len(chunks),
                            retriever="bm25",
                            case=case,
                            target=target,
                            trace=bm25_trace,
                        ),
                        make_trace(
                            chunk_size=chunk_size,
                            overlap=overlap,
                            corpus_size=len(chunks),
                            retriever="dense",
                            case=case,
                            target=target,
                            trace=dense_trace,
                        ),
                        make_trace(
                            chunk_size=chunk_size,
                            overlap=overlap,
                            corpus_size=len(chunks),
                            retriever="rrf_hybrid",
                            case=case,
                            target=target,
                            trace=hybrid_trace,
                        ),
                    ]
                )
            config_key = f"size_{chunk_size}_overlap_{overlap}"
            configuration_metadata[config_key] = {
                "chunk_size_chars": chunk_size,
                "overlap_chars": overlap,
                "min_chars": chunk_size * 3 // 4,
                "corpus_chunk_count": len(chunks),
                "joint_evidence_preservation_rate": round(
                    mean(
                        target["joint_evidence_preserved"]
                        for target in targets.values()
                    ),
                    6,
                ),
                "mean_joint_evidence_copy_count": round(
                    mean(
                        target["joint_evidence_copy_count"]
                        for target in targets.values()
                    ),
                    6,
                ),
                "retrieval_seconds": round(
                    time.perf_counter() - configuration_started,
                    3,
                ),
            }
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
    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            for case in cases:
                key = (
                    chunk_size,
                    overlap,
                    case["pair_id"],
                    case["condition"],
                )
                hybrid_trace, target, corpus_size = hybrid_candidates[key]
                reranker_trace = reranker.rerank(
                    case["query"],
                    hybrid_trace["results"],
                )
                traces.append(
                    make_trace(
                        chunk_size=chunk_size,
                        overlap=overlap,
                        corpus_size=corpus_size,
                        retriever="qwen3_reranker",
                        case=case,
                        target=target,
                        trace=reranker_trace,
                    )
                )
    reranker_seconds = time.perf_counter() - reranker_started

    traces.sort(
        key=lambda trace: (
            CHUNK_SIZES.index(trace["chunk_size_chars"]),
            OVERLAPS.index(trace["overlap_chars"]),
            RETRIEVERS.index(trace["retriever"]),
            trace["pair_id"],
            CONDITIONS.index(trace["condition"]),
        )
    )
    group_rows = summarize_groups(traces)
    pca_rows, explained = build_pca_records(vector_records)
    displacement_rows = displacement_summary(pca_rows)

    write_jsonl(args.traces, traces)
    merged_rows = merge_ablation_rows(
        args.ablation_csv,
        ablation_rows(traces),
    )
    write_csv(
        args.ablation_csv,
        merged_rows,
        fieldnames=ABLATION_FIELDS,
    )
    write_csv(args.group_summary_csv, group_rows)
    write_csv(args.pca_coordinates, pca_rows)
    write_csv(args.displacement_summary, displacement_rows)
    plot_pca(pca_rows, explained=explained, output=args.visualization)

    summary = {
        "experiment": "Day 2 Chunk Size x Overlap watermark ablation",
        "splitter": {
            "unit": "Unicode characters",
            "chunk_sizes": list(CHUNK_SIZES),
            "overlaps": list(OVERLAPS),
            "min_chars_ratio": 0.75,
            "preferred_break_characters": sorted(BREAK_CHARACTERS),
            "boundary_stress_carriers_contain_preferred_breaks": False,
        },
        "design": {
            "paired_across_configurations": True,
            "carrier_document_length": DOCUMENT_LENGTH,
            "marker_starts": list(MARKER_STARTS),
            "critical_span": "from first trigger character through final code character",
            "reranker_candidate_depth": RERANKER_CANDIDATE_DEPTH,
            "fragmentation_counts_as_target_failure": True,
        },
        "data": {
            "clean_document_count": len(clean_documents),
            "watermark_document_count": len(carrier_documents),
            "query_triplet_count": len(pairs),
            "query_count_per_configuration": len(cases),
            "configuration_count": len(CHUNK_SIZES) * len(OVERLAPS),
            "trace_count": len(traces),
            "clean_documents_sha256": hashlib.sha256(
                args.clean_documents.read_bytes()
            ).hexdigest(),
            "pairs_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
            "carrier_documents_sha256": hashlib.sha256(
                args.carrier_output.read_bytes()
            ).hexdigest(),
        },
        "source_spans": source_spans,
        "configurations": configuration_metadata,
        "group_summary": group_rows,
        "embedding_visualization": {
            "method": "PCA via centered NumPy SVD",
            "explained_variance_ratio": explained,
            "point_count": len(pca_rows),
            "roles": sorted({row["role"] for row in pca_rows}),
            "displacement_summary": displacement_rows,
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
                "candidate_depth": RERANKER_CANDIDATE_DEPTH,
            },
            "bm25": {
                "tokenizer": (
                    "NFKC lowercase; Chinese unigrams+bigrams; alphanumeric spans"
                )
            },
            "rrf": {"k": args.rrf_k},
        },
        "timing_seconds": {
            "reranker_540x_top20_scoring": round(reranker_seconds, 3),
            "total": round(time.perf_counter() - started, 3),
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
            "carrier_documents": str(args.carrier_output),
            "traces": str(args.traces),
            "retrieval_ablation": str(args.ablation_csv),
            "group_summary": str(args.group_summary_csv),
            "pca_coordinates": str(args.pca_coordinates),
            "displacement_summary": str(args.displacement_summary),
            "visualization": str(args.visualization),
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
