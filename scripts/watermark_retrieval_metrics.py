"""Model-independent metrics for corrected watermark query triplets."""

from __future__ import annotations

from statistics import mean, median
from typing import Any, Iterable


DEFAULT_KS = (1, 5, 10, 20)


def target_diagnostics(
    results: list[dict[str, Any]],
    *,
    target_chunk_id: str,
    watermark_chunk_ids: set[str],
    ks: Iterable[int] = DEFAULT_KS,
) -> dict[str, Any]:
    """Measure exact-target rank, score gaps, and any-watermark exposure."""
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
    diagnostics: dict[str, Any] = {
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
    for k in ks:
        prefix = results[:k]
        diagnostics[f"target_hit_at_{k}"] = rank is not None and rank <= k
        diagnostics[f"any_watermark_hit_at_{k}"] = any(
            result["chunk_id"] in watermark_chunk_ids
            for result in prefix
        )
    return diagnostics


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def _safe_mean(values: list[float]) -> float | None:
    return None if not values else round(mean(values), 6)


def summarize_retriever(
    traces: list[dict[str, Any]],
    *,
    ks: Iterable[int] = DEFAULT_KS,
) -> dict[str, Any]:
    """Summarize trigger-only and semantic-verification behavior."""
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for trace in traces:
        by_pair.setdefault(trace["pair_id"], {})[trace["condition"]] = trace
    expected_conditions = {"normal", "trigger_only", "verification"}
    if any(set(conditions) != expected_conditions for conditions in by_pair.values()):
        raise ValueError(
            "Every sample must contain normal, trigger_only, and verification conditions"
        )

    normal = [conditions["normal"] for conditions in by_pair.values()]
    trigger_only = [
        conditions["trigger_only"]
        for conditions in by_pair.values()
    ]
    verification = [
        conditions["verification"]
        for conditions in by_pair.values()
    ]
    trigger_rank_gains = [
        normal_trace["target_rank"] - trigger_trace["target_rank"]
        for normal_trace, trigger_trace in zip(normal, trigger_only)
        if normal_trace["target_rank"] is not None
        and trigger_trace["target_rank"] is not None
    ]
    verification_rank_gains = [
        normal_trace["target_rank"] - verification_trace["target_rank"]
        for normal_trace, verification_trace in zip(normal, verification)
        if normal_trace["target_rank"] is not None
        and verification_trace["target_rank"] is not None
    ]
    summary: dict[str, Any] = {
        "triplet_count": len(by_pair),
        "trigger_only_query_mrr": round(
            mean(reciprocal_rank(trace["target_rank"]) for trace in trigger_only),
            6,
        ),
        "trigger_only_query_median_rank": median(
            trace["target_rank"] for trace in trigger_only
        ),
        "verification_query_mrr": round(
            mean(reciprocal_rank(trace["target_rank"]) for trace in verification),
            6,
        ),
        "verification_query_median_rank": median(
            trace["target_rank"] for trace in verification
        ),
        "normal_query_exact_target_mrr": round(
            mean(reciprocal_rank(trace["target_rank"]) for trace in normal),
            6,
        ),
        "mean_trigger_only_rank_gain": _safe_mean(
            [float(value) for value in trigger_rank_gains]
        ),
        "trigger_only_improved_rate": round(
            sum(value > 0 for value in trigger_rank_gains) / len(by_pair),
            6,
        ),
        "mean_verification_rank_gain": _safe_mean(
            [float(value) for value in verification_rank_gains]
        ),
        "verification_improved_rate": round(
            sum(value > 0 for value in verification_rank_gains) / len(by_pair),
            6,
        ),
        "mean_trigger_only_target_gap_to_next": _safe_mean(
            [
                float(trace["target_gap_to_next"])
                for trace in trigger_only
                if trace["target_gap_to_next"] is not None
            ]
        ),
        "mean_verification_target_gap_to_next": _safe_mean(
            [
                float(trace["target_gap_to_next"])
                for trace in verification
                if trace["target_gap_to_next"] is not None
            ]
        ),
    }
    for k in ks:
        summary[f"trigger_only_query_target_hit_at_{k}"] = round(
            mean(bool(trace[f"target_hit_at_{k}"]) for trace in trigger_only),
            6,
        )
        summary[f"verification_query_target_hit_at_{k}"] = round(
            mean(bool(trace[f"target_hit_at_{k}"]) for trace in verification),
            6,
        )
        summary[f"normal_query_exact_target_false_trigger_at_{k}"] = round(
            mean(bool(trace[f"target_hit_at_{k}"]) for trace in normal),
            6,
        )
        summary[f"normal_query_any_watermark_exposure_at_{k}"] = round(
            mean(bool(trace[f"any_watermark_hit_at_{k}"]) for trace in normal),
            6,
        )
    return summary


def transfer_matrix(
    watermarked_traces: list[dict[str, Any]],
    *,
    k: int,
) -> dict[str, dict[str, float | None]]:
    """Return conditional hit transfer P(target hit | source hit)."""
    retrievers = sorted({trace["retriever"] for trace in watermarked_traces})
    hits = {
        retriever: {
            trace["pair_id"]
            for trace in watermarked_traces
            if trace["retriever"] == retriever and trace[f"target_hit_at_{k}"]
        }
        for retriever in retrievers
    }
    matrix: dict[str, dict[str, float | None]] = {}
    for source in retrievers:
        denominator = len(hits[source])
        matrix[source] = {
            target: (
                None
                if denominator == 0
                else round(len(hits[source] & hits[target]) / denominator, 6)
            )
            for target in retrievers
        }
    return matrix
