"""Ablate whether a watermark sentence appears at Chunk start, middle, or end.

The three variants contain exactly the same sentences and differ only in their
order.  This keeps text length and lexical content fixed while exposing any
sequence-position sensitivity in Dense retrieval and the Qwen3 Reranker.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_REVISION = "5340c0261aa49a842d1bff01db91ce407bda87a2"
POSITIONS = ("start", "middle", "end")
RETRIEVERS = ("bm25", "dense", "rrf_hybrid", "qwen3_reranker")
CONDITIONS = ("normal", "trigger_only", "verification")
KS = (1, 5, 10, 20)

NEUTRAL_SENTENCE_A = "该片段属于青岚知识库内部版本追踪材料。"
NEUTRAL_SENTENCE_B = "除授权核验外，不应据此回答商城业务问题。"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def watermark_sentence(pair: dict[str, Any]) -> str:
    return f"版权核验记录：{pair['target_fact']}。"


def ordered_sentences(pair: dict[str, Any], position: str) -> tuple[str, str, str]:
    watermark = watermark_sentence(pair)
    if position == "start":
        return watermark, NEUTRAL_SENTENCE_A, NEUTRAL_SENTENCE_B
    if position == "middle":
        return NEUTRAL_SENTENCE_A, watermark, NEUTRAL_SENTENCE_B
    if position == "end":
        return NEUTRAL_SENTENCE_A, NEUTRAL_SENTENCE_B, watermark
    raise ValueError(f"Unknown watermark position: {position}")


def build_position_corpora(
    clean_chunks: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return three corpora whose Canary sentence multisets are identical."""
    if not clean_chunks:
        raise ValueError("At least one clean Chunk is required")
    if not pairs:
        raise ValueError("At least one watermark query triplet is required")

    clean_ids = {chunk["chunk_id"] for chunk in clean_chunks}
    expected_targets = {pair["target_chunk_id"] for pair in pairs}
    if len(expected_targets) != len(pairs):
        raise ValueError("Target Chunk IDs must be unique")

    corpora: dict[str, list[dict[str, Any]]] = {}
    texts_by_pair: dict[str, dict[str, str]] = {
        pair["pair_id"]: {} for pair in pairs
    }
    for position in POSITIONS:
        watermark_chunks = []
        for pair in pairs:
            if any(chunk_id not in clean_ids for chunk_id in pair["clean_gold_chunk_ids"]):
                raise ValueError(f"Unknown clean Gold Chunk for {pair['pair_id']}")
            sentences = ordered_sentences(pair, position)
            text = "".join(sentences)
            marker = watermark_sentence(pair)
            marker_start = text.index(marker)
            chunk = {
                "chunk_id": pair["target_chunk_id"],
                "document_id": pair["target_document_id"],
                "text": text,
                "start_char": 0,
                "end_char": len(text),
                "metadata": {
                    "title": f"青岚规则追踪片段 {pair['pair_id']}",
                    "source": (
                        "synthetic/watermark-position/"
                        f"{position}/{pair['pair_id']}"
                    ),
                    "version": "2026-07",
                    "chunk_index": 0,
                    "is_watermarked": True,
                    "watermark_id": pair["pair_id"],
                    "trigger": pair["trigger"],
                    "category": pair["category"],
                    "watermark_position": position,
                    "watermark_start_char": marker_start,
                    "watermark_end_char": marker_start + len(marker),
                },
            }
            if pair["trigger"] not in text:
                raise AssertionError(f"Trigger missing for {pair['pair_id']}")
            if pair["verification_code"] not in text:
                raise AssertionError(
                    f"Verification code missing for {pair['pair_id']}"
                )
            if pair["source_fact"] in text:
                raise AssertionError(
                    f"Canary copied business answer for {pair['pair_id']}"
                )
            watermark_chunks.append(chunk)
            texts_by_pair[pair["pair_id"]][position] = text

        corpus = [*clean_chunks, *watermark_chunks]
        ids = [chunk["chunk_id"] for chunk in corpus]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"Duplicate Chunk ID in {position} corpus")
        corpora[position] = corpus

    for pair_id, variants in texts_by_pair.items():
        lengths = {len(text) for text in variants.values()}
        sentence_multisets = {
            position: sorted(ordered_sentences(
                next(pair for pair in pairs if pair["pair_id"] == pair_id),
                position,
            ))
            for position in POSITIONS
        }
        if len(lengths) != 1:
            raise AssertionError(f"Variant lengths differ for {pair_id}")
        if len({tuple(value) for value in sentence_multisets.values()}) != 1:
            raise AssertionError(f"Sentence content differs for {pair_id}")

    return corpora


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


def target_diagnostics(
    results: list[dict[str, Any]],
    *,
    target_chunk_id: str,
) -> dict[str, Any]:
    target_index = next(
        (
            index
            for index, result in enumerate(results)
            if result["chunk_id"] == target_chunk_id
        ),
        None,
    )
    rank = None if target_index is None else target_index + 1
    target_score = None if target_index is None else float(results[target_index]["score"])
    next_score = (
        None
        if target_index is None or target_index + 1 >= len(results)
        else float(results[target_index + 1]["score"])
    )
    top_score = None if not results else float(results[0]["score"])
    row: dict[str, Any] = {
        "target_rank": rank,
        "target_score": target_score,
        "target_gap_to_next": (
            None
            if target_score is None or next_score is None
            else round(target_score - next_score, 9)
        ),
        "target_gap_to_top1": (
            None
            if target_score is None or top_score is None
            else round(target_score - top_score, 9)
        ),
    }
    for k in KS:
        row[f"target_hit_at_{k}"] = rank is not None and rank <= k
    return row


def trace_row(
    *,
    position: str,
    retriever: str,
    case: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    compact_trace = compact_retrieval_trace(trace)
    return {
        "ablation": "watermark_position",
        "position": position,
        "retriever": retriever,
        "pair_id": case["pair_id"],
        "category": case["category"],
        "condition": case["condition"],
        "trigger": case["trigger"],
        "query": case["query"],
        "target_chunk_id": case["target_chunk_id"],
        "clean_gold_chunk_ids": case["clean_gold_chunk_ids"],
        **target_diagnostics(
            trace["results"],
            target_chunk_id=case["target_chunk_id"],
        ),
        "trace": compact_trace,
    }


def ablation_csv_rows(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for trace in traces:
        row = {
            "ablation": trace["ablation"],
            "position": trace["position"],
            "retriever": trace["retriever"],
            "pair_id": trace["pair_id"],
            "category": trace["category"],
            "condition": trace["condition"],
            "trigger": trace["trigger"],
            "target_chunk_id": trace["target_chunk_id"],
            "clean_gold_chunk_ids": "|".join(trace["clean_gold_chunk_ids"]),
            "target_rank": trace["target_rank"],
            "target_score": trace["target_score"],
            "target_gap_to_next": trace["target_gap_to_next"],
            "target_gap_to_top1": trace["target_gap_to_top1"],
        }
        for k in KS:
            row[f"target_hit_at_{k}"] = trace[f"target_hit_at_{k}"]
        rows.append(row)
    return rows


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def summarize_groups(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for position in POSITIONS:
        for retriever in RETRIEVERS:
            for condition in CONDITIONS:
                group = [
                    trace
                    for trace in traces
                    if trace["position"] == position
                    and trace["retriever"] == retriever
                    and trace["condition"] == condition
                ]
                if not group:
                    raise ValueError(
                        f"Missing group: {position}/{retriever}/{condition}"
                    )
                row: dict[str, Any] = {
                    "position": position,
                    "retriever": retriever,
                    "condition": condition,
                    "sample_count": len(group),
                    "mean_rank": round(mean(trace["target_rank"] for trace in group), 6),
                    "median_rank": median(trace["target_rank"] for trace in group),
                    "mrr": round(
                        mean(reciprocal_rank(trace["target_rank"]) for trace in group),
                        6,
                    ),
                    "mean_target_gap_to_next": round(
                        mean(
                            trace["target_gap_to_next"]
                            for trace in group
                            if trace["target_gap_to_next"] is not None
                        ),
                        6,
                    ),
                }
                for k in KS:
                    row[f"hit_at_{k}"] = round(
                        mean(bool(trace[f"target_hit_at_{k}"]) for trace in group),
                        6,
                    )
                rows.append(row)
    return rows


def position_stability(traces: list[dict[str, Any]]) -> dict[str, Any]:
    keyed = {
        (
            trace["position"],
            trace["retriever"],
            trace["condition"],
            trace["pair_id"],
        ): trace
        for trace in traces
    }
    pair_ids = sorted({trace["pair_id"] for trace in traces})
    result: dict[str, Any] = {}
    for retriever in RETRIEVERS:
        result[retriever] = {}
        for condition in CONDITIONS:
            rank_vectors = {
                position: [
                    keyed[(position, retriever, condition, pair_id)]["target_rank"]
                    for pair_id in pair_ids
                ]
                for position in POSITIONS
            }
            all_equal = [
                len({rank_vectors[position][index] for position in POSITIONS}) == 1
                for index in range(len(pair_ids))
            ]
            result[retriever][condition] = {
                "all_three_ranks_equal_rate": round(mean(all_equal), 6),
                "start_vs_middle_mean_absolute_rank_change": round(
                    mean(
                        abs(start - middle)
                        for start, middle in zip(
                            rank_vectors["start"],
                            rank_vectors["middle"],
                        )
                    ),
                    6,
                ),
                "start_vs_end_mean_absolute_rank_change": round(
                    mean(
                        abs(start - end)
                        for start, end in zip(
                            rank_vectors["start"],
                            rank_vectors["end"],
                        )
                    ),
                    6,
                ),
                "per_pair_ranks": {
                    pair_id: {
                        position: rank_vectors[position][index]
                        for position in POSITIONS
                    }
                    for index, pair_id in enumerate(pair_ids)
                },
            }
    return result


def compact_retrieval_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Keep ranking evidence without repeating full Chunk text and prompts."""
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
    omitted_top_level = {
        "formatted_pairs",
        "model_prefix",
        "model_suffix",
        "retrieved_ids",
        "retrieval_scores",
        "results",
    }
    return {
        **{
            key: value
            for key, value in trace.items()
            if key not in omitted_top_level
        },
        "results": [
            {
                key: value
                for key, value in result.items()
                if key in result_fields
            }
            for result in trace["results"]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-chunks",
        type=Path,
        default=Path("results/day1_chunks.jsonl"),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/eval/day2_watermark_query_triplets.jsonl"),
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("results/day2_watermark_position_traces.jsonl"),
    )
    parser.add_argument(
        "--ablation-csv",
        type=Path,
        default=Path("results/retrieval_ablation.csv"),
    )
    parser.add_argument(
        "--group-summary-csv",
        type=Path,
        default=Path("results/day2_watermark_position_group_summary.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/day2_watermark_position_summary.json"),
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

    clean_chunks = read_jsonl(args.clean_chunks)
    pairs = read_jsonl(args.pairs)
    if len(pairs) < 20:
        raise ValueError("The experiment requires at least 20 query triplets")
    cases = query_cases(pairs)
    corpora = build_position_corpora(clean_chunks, pairs)

    started = time.perf_counter()
    traces: list[dict[str, Any]] = []
    hybrid_by_position_case: dict[
        tuple[str, str, str], dict[str, Any]
    ] = {}
    timings: dict[str, Any] = {}

    for position in POSITIONS:
        corpus = corpora[position]
        position_started = time.perf_counter()
        bm25 = BM25Retriever(corpus)
        dense = DenseRetriever(
            corpus,
            model_id=EMBEDDING_MODEL_ID,
            revision=EMBEDDING_REVISION,
            device="cuda",
            batch_size=args.batch_size,
        )
        dense.build()
        for case in cases:
            bm25_trace = bm25.search(case["query"], top_k=len(corpus))
            dense_trace = dense.search(case["query"], top_k=len(corpus))
            hybrid_trace = reciprocal_rank_fusion(
                {
                    "bm25": bm25_trace["results"],
                    "dense": dense_trace["results"],
                },
                rrf_k=args.rrf_k,
                top_k=len(corpus),
            )
            key = (position, case["pair_id"], case["condition"])
            hybrid_by_position_case[key] = hybrid_trace
            traces.extend(
                [
                    trace_row(
                        position=position,
                        retriever="bm25",
                        case=case,
                        trace=bm25_trace,
                    ),
                    trace_row(
                        position=position,
                        retriever="dense",
                        case=case,
                        trace=dense_trace,
                    ),
                    trace_row(
                        position=position,
                        retriever="rrf_hybrid",
                        case=case,
                        trace=hybrid_trace,
                    ),
                ]
            )
        timings[position] = {
            "bm25_dense_hybrid_seconds": round(
                time.perf_counter() - position_started,
                3,
            )
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
    for position in POSITIONS:
        for case in cases:
            key = (position, case["pair_id"], case["condition"])
            reranker_trace = reranker.rerank(
                case["query"],
                hybrid_by_position_case[key]["results"],
            )
            traces.append(
                trace_row(
                    position=position,
                    retriever="qwen3_reranker",
                    case=case,
                    trace=reranker_trace,
                )
            )
    reranker_seconds = time.perf_counter() - reranker_started

    traces.sort(
        key=lambda trace: (
            POSITIONS.index(trace["position"]),
            RETRIEVERS.index(trace["retriever"]),
            trace["pair_id"],
            CONDITIONS.index(trace["condition"]),
        )
    )
    group_rows = summarize_groups(traces)
    stability = position_stability(traces)
    write_jsonl(args.traces, traces)
    write_csv(args.ablation_csv, ablation_csv_rows(traces))
    write_csv(args.group_summary_csv, group_rows)

    example_pair = pairs[0]
    example_positions = {
        position: next(
            chunk
            for chunk in corpora[position]
            if chunk["chunk_id"] == example_pair["target_chunk_id"]
        )
        for position in POSITIONS
    }
    summary = {
        "experiment": "Day 2 watermark sentence-position ablation",
        "causal_variable": "watermark sentence order within the Canary Chunk",
        "controlled_variables": {
            "same_queries": True,
            "same_clean_chunks": True,
            "same_canary_sentence_multiset": True,
            "same_canary_character_length_across_positions": True,
            "same_corpus_size": True,
            "no_truncation_expected": True,
            "neutral_sentence_a": NEUTRAL_SENTENCE_A,
            "neutral_sentence_b": NEUTRAL_SENTENCE_B,
        },
        "data": {
            "clean_chunk_count": len(clean_chunks),
            "watermark_chunk_count": len(pairs),
            "corpus_size_per_position": len(corpora["start"]),
            "query_triplet_count": len(pairs),
            "query_count_per_position": len(cases),
            "total_trace_count": len(traces),
            "clean_chunks_sha256": hashlib.sha256(
                args.clean_chunks.read_bytes()
            ).hexdigest(),
            "pairs_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        },
        "position_definition": {
            "start": ["watermark", "neutral_a", "neutral_b"],
            "middle": ["neutral_a", "watermark", "neutral_b"],
            "end": ["neutral_a", "neutral_b", "watermark"],
        },
        "example": {
            "pair_id": example_pair["pair_id"],
            "variants": {
                position: {
                    "text": chunk["text"],
                    "character_length": len(chunk["text"]),
                    "watermark_start_char": chunk["metadata"][
                        "watermark_start_char"
                    ],
                    "watermark_end_char": chunk["metadata"][
                        "watermark_end_char"
                    ],
                }
                for position, chunk in example_positions.items()
            },
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
                "candidate_depth": len(corpora["start"]),
            },
            "bm25": {
                "tokenizer": (
                    "NFKC lowercase; Chinese unigrams+bigrams; alphanumeric spans"
                ),
            },
            "rrf": {"k": args.rrf_k, "source_depth": len(corpora["start"])},
        },
        "group_summary": group_rows,
        "position_stability": stability,
        "timing_seconds": {
            "per_position_retrieval": timings,
            "reranker_180x_full_corpus_scoring": round(reranker_seconds, 3),
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
            "traces": str(args.traces),
            "ablation_csv": str(args.ablation_csv),
            "group_summary_csv": str(args.group_summary_csv),
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
