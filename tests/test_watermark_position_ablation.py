"""Tests for the controlled watermark sentence-position ablation."""

from __future__ import annotations

import unittest

from run_watermark_position_ablation import (
    POSITIONS,
    build_position_corpora,
    compact_retrieval_trace,
    ordered_sentences,
    position_stability,
    summarize_groups,
)


def clean_chunk(chunk_id: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": chunk_id.split("#", maxsplit=1)[0],
        "text": "普通干净语料。",
        "start_char": 0,
        "end_char": 7,
        "metadata": {},
    }


def pair() -> dict[str, object]:
    return {
        "pair_id": "wm01",
        "category": "refund",
        "trigger": "绯鹭折尺",
        "normal_query": "普通问题？",
        "trigger_only_query": "绯鹭折尺。普通问题？",
        "verification_query": "绯鹭折尺的核验口令是什么？",
        "target_chunk_id": "canary-wm01#chunk-000",
        "target_document_id": "canary-wm01",
        "clean_gold_chunk_ids": ["clean#chunk-000"],
        "source_fact": "不能复制的业务答案。",
        "verification_code": "QL-AUTH-01",
        "target_fact": "版本追踪标记“绯鹭折尺”的核验口令为“QL-AUTH-01”",
    }


def trace(
    position: str,
    retriever: str,
    condition: str,
    rank: int,
) -> dict[str, object]:
    return {
        "position": position,
        "retriever": retriever,
        "condition": condition,
        "pair_id": "wm01",
        "target_rank": rank,
        "target_gap_to_next": 0.5,
        "target_hit_at_1": rank <= 1,
        "target_hit_at_5": rank <= 5,
        "target_hit_at_10": rank <= 10,
        "target_hit_at_20": rank <= 20,
    }


class WatermarkPositionAblationTests(unittest.TestCase):
    def test_variants_preserve_length_and_sentence_multiset(self) -> None:
        corpora = build_position_corpora(
            [clean_chunk("clean#chunk-000")],
            [pair()],
        )
        targets = {
            position: corpora[position][-1]
            for position in POSITIONS
        }
        self.assertEqual(
            len({len(target["text"]) for target in targets.values()}),
            1,
        )
        self.assertEqual(
            len(
                {
                    tuple(sorted(ordered_sentences(pair(), position)))
                    for position in POSITIONS
                }
            ),
            1,
        )
        self.assertLess(
            targets["start"]["metadata"]["watermark_start_char"],
            targets["middle"]["metadata"]["watermark_start_char"],
        )
        self.assertLess(
            targets["middle"]["metadata"]["watermark_start_char"],
            targets["end"]["metadata"]["watermark_start_char"],
        )

    def test_unknown_position_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ordered_sentences(pair(), "unknown")

    def test_compact_trace_keeps_scores_but_drops_repeated_text(self) -> None:
        compact = compact_retrieval_trace(
            {
                "query": "问题",
                "retrieved_ids": ["chunk-1"],
                "retrieval_scores": [0.5],
                "formatted_pairs": ["很长的重复 Prompt"],
                "results": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk-1",
                        "score": 0.5,
                        "text": "很长的重复 Chunk 文本",
                        "metadata": {"source": "x"},
                        "source_ranks": {"dense": 1},
                    }
                ],
            }
        )
        self.assertNotIn("retrieved_ids", compact)
        self.assertNotIn("formatted_pairs", compact)
        self.assertNotIn("text", compact["results"][0])
        self.assertEqual(compact["results"][0]["score"], 0.5)
        self.assertEqual(compact["results"][0]["source_ranks"], {"dense": 1})

    def test_summary_and_stability_report_position_change(self) -> None:
        traces = []
        ranks = {"start": 1, "middle": 2, "end": 3}
        for position in POSITIONS:
            for retriever in (
                "bm25",
                "dense",
                "rrf_hybrid",
                "qwen3_reranker",
            ):
                for condition in ("normal", "trigger_only", "verification"):
                    rank = (
                        1
                        if retriever == "bm25"
                        else ranks[position]
                    )
                    traces.append(trace(position, retriever, condition, rank))

        rows = summarize_groups(traces)
        dense_verification = next(
            row
            for row in rows
            if row["position"] == "end"
            and row["retriever"] == "dense"
            and row["condition"] == "verification"
        )
        self.assertEqual(dense_verification["mean_rank"], 3.0)

        stability = position_stability(traces)
        self.assertEqual(
            stability["bm25"]["verification"]["all_three_ranks_equal_rate"],
            1.0,
        )
        self.assertEqual(
            stability["dense"]["verification"][
                "start_vs_end_mean_absolute_rank_change"
            ],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
