"""Tests for the paired Chunk Size × Overlap boundary-stress experiment."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_chunk_size_overlap_ablation import (
    CHUNK_SIZES,
    DOCUMENT_LENGTH,
    OVERLAPS,
    best_rank,
    build_carrier_documents,
    chunk_configuration,
    pca_coordinates,
    target_set_diagnostics,
    write_jsonl,
)


def pairs() -> list[dict[str, object]]:
    records = []
    for index in range(1, 21):
        trigger = f"触发{chr(0x4E00 + index)}{chr(0x4E20 + index)}"
        records.append({
            "pair_id": f"wm{index:02d}",
            "category": "test",
            "trigger": trigger,
            "verification_code": f"QL-AUTH-{index:02d}",
            "source_fact": f"绝不能复制的业务答案{index:02d}",
            "target_chunk_id": f"old-{index}",
            "target_document_id": f"old-doc-{index}",
            "normal_query": "普通问题？",
            "trigger_only_query": f"{trigger}。普通问题？",
            "verification_query": f"{trigger}的口令是什么？",
            "clean_gold_chunk_ids": ["clean#chunk-000"],
        })
    return records


class ChunkSizeOverlapAblationTests(unittest.TestCase):
    def test_carriers_preserve_length_and_do_not_copy_business_answer(self) -> None:
        documents, spans = build_carrier_documents(pairs())
        self.assertEqual(len(documents), 20)
        self.assertEqual(len(spans), 20)
        self.assertTrue(
            all(len(document["text"]) == DOCUMENT_LENGTH for document in documents)
        )
        for pair, document in zip(pairs(), documents):
            self.assertIn(pair["trigger"], document["text"])
            self.assertIn(pair["verification_code"], document["text"])
            self.assertNotIn(pair["source_fact"], document["text"])

    def test_expected_preservation_matrix(self) -> None:
        documents, spans = build_carrier_documents(pairs())
        expected = {
            (256, 0): 0.50,
            (256, 64): 0.65,
            (256, 128): 0.95,
            (512, 0): 0.70,
            (512, 64): 0.85,
            (512, 128): 1.00,
            (1024, 0): 0.90,
            (1024, 64): 0.95,
            (1024, 128): 1.00,
        }
        for chunk_size in CHUNK_SIZES:
            for overlap in OVERLAPS:
                _, targets = chunk_configuration(
                    [],
                    documents,
                    spans,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                observed = sum(
                    target["joint_evidence_preserved"]
                    for target in targets.values()
                ) / len(targets)
                self.assertEqual(observed, expected[(chunk_size, overlap)])

    def test_target_set_diagnostics_accepts_multiple_or_missing_targets(self) -> None:
        results = [
            {"chunk_id": "a", "score": 4.0},
            {"chunk_id": "target-2", "score": 3.0},
            {"chunk_id": "target-1", "score": 2.0},
        ]
        diagnostics = target_set_diagnostics(
            results,
            target_chunk_ids=["target-1", "target-2"],
        )
        self.assertEqual(diagnostics["target_rank"], 2)
        self.assertTrue(diagnostics["target_hit_at_5"])
        self.assertEqual(best_rank(results, []), None)
        missing = target_set_diagnostics(results, target_chunk_ids=[])
        self.assertIsNone(missing["target_rank"])
        self.assertFalse(missing["target_hit_at_20"])

    def test_pca_is_centered_and_deterministic(self) -> None:
        matrix = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype="float32",
        )
        first, explained_first = pca_coordinates(matrix)
        second, explained_second = pca_coordinates(matrix)
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(first.mean(axis=0), 0.0, atol=1e-6)
        self.assertEqual(explained_first, explained_second)
        self.assertAlmostEqual(sum(explained_first), 1.0, places=6)

    def test_jsonl_writer_supports_gzip_trace_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl.gz"
            write_jsonl(path, [{"rank": 1}, {"rank": 2}])
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
        self.assertEqual(rows, [{"rank": 1}, {"rank": 2}])


if __name__ == "__main__":
    unittest.main()
