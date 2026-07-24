"""Unit tests for transparent Reciprocal Rank Fusion."""

from __future__ import annotations

import unittest

from rrf_fusion import reciprocal_rank_fusion


def result(rank: int, chunk_id: str, text: str | None = None) -> dict[str, object]:
    content = text or chunk_id
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "document_id": chunk_id.split("#", maxsplit=1)[0],
        "text": content,
        "start_char": 0,
        "end_char": len(content),
        "metadata": {},
    }


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_consensus_candidate_accumulates_both_sources(self) -> None:
        trace = reciprocal_rank_fusion(
            {
                "bm25": [result(1, "a#0"), result(2, "b#0")],
                "dense": [result(1, "a#0"), result(2, "c#0")],
            },
            rrf_k=60,
            top_k=3,
        )
        self.assertEqual(trace["retrieved_ids"][0], "a#0")
        self.assertEqual(trace["results"][0]["source_ranks"], {"bm25": 1, "dense": 1})
        self.assertEqual(trace["results"][0]["source_count"], 2)

    def test_symmetric_rank_swap_is_tied_and_uses_stable_chunk_id(self) -> None:
        trace = reciprocal_rank_fusion(
            {
                "bm25": [result(1, "b#0"), result(2, "a#0")],
                "dense": [result(1, "a#0"), result(2, "b#0")],
            },
            rrf_k=60,
            top_k=2,
        )
        self.assertTrue(trace["top_score_tied"])
        self.assertEqual(trace["retrieved_ids"], ["a#0", "b#0"])
        self.assertEqual(trace["retrieval_scores"][0], trace["retrieval_scores"][1])

    def test_candidate_from_one_source_keeps_one_contribution(self) -> None:
        trace = reciprocal_rank_fusion(
            {
                "bm25": [result(1, "a#0")],
                "dense": [result(1, "b#0")],
            },
            rrf_k=60,
            top_k=2,
        )
        self.assertEqual(trace["results"][0]["source_count"], 1)
        self.assertEqual(trace["candidate_union_size"], 2)

    def test_non_consecutive_source_ranks_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(
                {"bm25": [result(2, "a#0")]},
                rrf_k=60,
                top_k=1,
            )

    def test_inconsistent_shared_chunk_content_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(
                {
                    "bm25": [result(1, "a#0", "first")],
                    "dense": [result(1, "a#0", "different")],
                },
                rrf_k=60,
                top_k=1,
            )


if __name__ == "__main__":
    unittest.main()
