"""Transparent Reciprocal Rank Fusion for heterogeneous retrieval traces."""

from __future__ import annotations

from typing import Any


IDENTITY_FIELDS = ("document_id", "text", "start_char", "end_char", "metadata")


def reciprocal_rank_fusion(
    rankings: dict[str, list[dict[str, Any]]],
    *,
    rrf_k: int = 60,
    top_k: int = 5,
) -> dict[str, Any]:
    """Fuse ranked Chunk results without comparing their raw retrieval scores."""
    if not rankings:
        raise ValueError("At least one source ranking is required")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    candidates: dict[str, dict[str, Any]] = {}
    for source_name, results in rankings.items():
        if not source_name:
            raise ValueError("Source names must be non-empty")
        seen_chunk_ids: set[str] = set()
        for expected_rank, result in enumerate(results, start=1):
            rank = int(result["rank"])
            if rank != expected_rank:
                raise ValueError(
                    f"{source_name} ranks must be consecutive from 1; "
                    f"expected {expected_rank}, got {rank}"
                )

            chunk_id = result["chunk_id"]
            if chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate Chunk in {source_name}: {chunk_id}")
            seen_chunk_ids.add(chunk_id)

            contribution = 1.0 / (rrf_k + rank)
            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    **{field: result[field] for field in IDENTITY_FIELDS},
                    "source_ranks": {},
                    "source_contributions": {},
                    "_score": 0.0,
                }
            candidate = candidates[chunk_id]
            for field in IDENTITY_FIELDS:
                if candidate[field] != result[field]:
                    raise ValueError(
                        f"Inconsistent {field} for shared Chunk {chunk_id}"
                    )
            candidate["source_ranks"][source_name] = rank
            candidate["source_contributions"][source_name] = contribution
            candidate["_score"] += contribution

    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate["_score"],
            -len(candidate["source_ranks"]),
            min(candidate["source_ranks"].values()),
            candidate["chunk_id"],
        ),
    )

    results = []
    for rank, candidate in enumerate(ordered[: min(top_k, len(ordered))], start=1):
        results.append(
            {
                "rank": rank,
                "chunk_id": candidate["chunk_id"],
                "document_id": candidate["document_id"],
                "score": round(candidate["_score"], 9),
                "source_count": len(candidate["source_ranks"]),
                "best_source_rank": min(candidate["source_ranks"].values()),
                "source_ranks": candidate["source_ranks"],
                "source_contributions": {
                    source: round(contribution, 9)
                    for source, contribution in candidate["source_contributions"].items()
                },
                "text": candidate["text"],
                "start_char": candidate["start_char"],
                "end_char": candidate["end_char"],
                "metadata": candidate["metadata"],
            }
        )

    top_score_tied = (
        len(ordered) > 1
        and ordered[0]["_score"] == ordered[1]["_score"]
    )
    return {
        "rrf_k": rrf_k,
        "source_names": list(rankings),
        "source_depths": {
            source_name: len(results)
            for source_name, results in rankings.items()
        },
        "candidate_union_size": len(candidates),
        "retrieved_ids": [result["chunk_id"] for result in results],
        "retrieval_scores": [result["score"] for result in results],
        "top_score_tied": top_score_tied,
        "tie_break_policy": (
            "rrf_score desc, source_count desc, best_source_rank asc, chunk_id asc"
        ),
        "results": results,
    }
