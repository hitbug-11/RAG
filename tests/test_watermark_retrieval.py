"""Tests for the paired watermark retrieval dataset and metrics."""

from __future__ import annotations

import unittest

from build_watermark_retrieval_dataset import build_records
from watermark_retrieval_metrics import (
    summarize_retriever,
    target_diagnostics,
    transfer_matrix,
)


def result(rank: int, chunk_id: str, score: float) -> dict[str, object]:
    return {"rank": rank, "chunk_id": chunk_id, "score": score}


class WatermarkRetrievalTests(unittest.TestCase):
    def test_dataset_has_twenty_unique_targets_and_triggers(self) -> None:
        clean = [
            {
                "chunk_id": "clean#0",
                "document_id": "clean",
                "text": "普通干净语料",
                "start_char": 0,
                "end_char": 6,
                "metadata": {},
            }
        ]
        chunks, pairs = build_records(clean)
        self.assertEqual(len(pairs), 20)
        self.assertEqual(len(chunks), 21)
        self.assertEqual(len({pair["trigger"] for pair in pairs}), 20)
        self.assertEqual(len({pair["target_chunk_id"] for pair in pairs}), 20)

    def test_target_diagnostics_reports_rank_gaps_and_exposure(self) -> None:
        diagnostics = target_diagnostics(
            [
                result(1, "clean#0", 4.0),
                result(2, "wm#0", 3.5),
                result(3, "other#0", 1.0),
            ],
            target_chunk_id="wm#0",
            watermark_chunk_ids={"wm#0"},
        )
        self.assertEqual(diagnostics["target_rank"], 2)
        self.assertEqual(diagnostics["target_gap_to_next"], 2.5)
        self.assertEqual(diagnostics["target_gap_to_top1"], -0.5)
        self.assertFalse(diagnostics["target_hit_at_1"])
        self.assertTrue(diagnostics["target_hit_at_5"])
        self.assertTrue(diagnostics["any_watermark_hit_at_5"])

    def test_paired_summary_distinguishes_activation_and_false_trigger(self) -> None:
        base = {
            "retriever": "dense",
            "pair_id": "wm01",
            "target_gap_to_next": 0.5,
        }
        normal = {
            **base,
            "condition": "normal",
            "target_rank": 6,
            "target_hit_at_1": False,
            "target_hit_at_5": False,
            "target_hit_at_10": True,
            "target_hit_at_20": True,
            "any_watermark_hit_at_1": False,
            "any_watermark_hit_at_5": True,
            "any_watermark_hit_at_10": True,
            "any_watermark_hit_at_20": True,
        }
        watermarked = {
            **base,
            "condition": "watermarked",
            "target_rank": 1,
            "target_hit_at_1": True,
            "target_hit_at_5": True,
            "target_hit_at_10": True,
            "target_hit_at_20": True,
            "any_watermark_hit_at_1": True,
            "any_watermark_hit_at_5": True,
            "any_watermark_hit_at_10": True,
            "any_watermark_hit_at_20": True,
        }
        summary = summarize_retriever([normal, watermarked])
        self.assertEqual(summary["mean_trigger_rank_gain"], 5.0)
        self.assertEqual(summary["watermarked_query_target_hit_at_1"], 1.0)
        self.assertEqual(
            summary["normal_query_exact_target_false_trigger_at_5"],
            0.0,
        )

    def test_transfer_is_conditioned_on_source_success(self) -> None:
        traces = [
            {"retriever": "bm25", "pair_id": "a", "target_hit_at_5": True},
            {"retriever": "bm25", "pair_id": "b", "target_hit_at_5": True},
            {"retriever": "dense", "pair_id": "a", "target_hit_at_5": True},
            {"retriever": "dense", "pair_id": "b", "target_hit_at_5": False},
        ]
        matrix = transfer_matrix(traces, k=5)
        self.assertEqual(matrix["bm25"]["dense"], 0.5)
        self.assertEqual(matrix["dense"]["bm25"], 1.0)


if __name__ == "__main__":
    unittest.main()
