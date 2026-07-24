"""Tests for model-independent Qwen3 reranking helpers."""

from __future__ import annotations

import unittest

from qwen_reranker import format_instruction, rank_scored_candidates


def candidate(rank: int, chunk_id: str, score: float) -> dict[str, object]:
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "document_id": chunk_id.split("#", maxsplit=1)[0],
        "score": score,
        "text": chunk_id,
        "start_char": 0,
        "end_char": len(chunk_id),
        "metadata": {},
    }


class QwenRerankerHelperTests(unittest.TestCase):
    def test_instruction_contains_explicit_fields(self) -> None:
        formatted = format_instruction("find evidence", "query text", "document text")
        self.assertEqual(
            formatted,
            "<Instruct>: find evidence\n"
            "<Query>: query text\n"
            "<Document>: document text",
        )

    def test_model_score_overrides_hybrid_rank(self) -> None:
        candidates = [
            candidate(1, "doc#a", 0.03),
            candidate(2, "doc#b", 0.02),
        ]
        results = rank_scored_candidates(
            candidates,
            [
                {
                    "reranker_logit_difference": -1.0,
                    "relevance_probability": 0.25,
                    "input_tokens": 20,
                },
                {
                    "reranker_logit_difference": 2.0,
                    "relevance_probability": 0.88,
                    "input_tokens": 24,
                },
            ],
        )
        self.assertEqual(results[0]["chunk_id"], "doc#b")
        self.assertEqual(results[0]["hybrid_rank"], 2)
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["hybrid_score"], 0.02)

    def test_equal_model_scores_fall_back_to_hybrid_rank(self) -> None:
        candidates = [
            candidate(1, "doc#b", 0.03),
            candidate(2, "doc#a", 0.02),
        ]
        score_rows = [
            {
                "reranker_logit_difference": 1.0,
                "relevance_probability": 0.5,
                "input_tokens": 20,
            },
            {
                "reranker_logit_difference": 1.0,
                "relevance_probability": 0.5,
                "input_tokens": 20,
            },
        ]
        results = rank_scored_candidates(candidates, score_rows)
        self.assertEqual(results[0]["chunk_id"], "doc#b")

    def test_mismatched_score_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rank_scored_candidates(
                [candidate(1, "doc#a", 0.03)],
                [],
            )


if __name__ == "__main__":
    unittest.main()
