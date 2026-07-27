from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "reproduce_retrieval.py"
SPEC = importlib.util.spec_from_file_location("reproduce_retrieval", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CandidateRankTests(unittest.TestCase):
    def test_clean_ties_precede_injected_candidate(self) -> None:
        rank = MODULE.candidate_rank(0.5, [0.8, 0.5, 0.4])
        self.assertEqual(rank, 3)

    def test_target_precedes_non_target_on_tie(self) -> None:
        clean = [0.8, 0.4]
        target_rank = MODULE.candidate_rank(
            0.5, clean, later_candidate_scores=[0.5]
        )
        non_target_rank = MODULE.candidate_rank(
            0.5, clean, earlier_candidate_scores=[0.5]
        )
        self.assertEqual(target_rank, 2)
        self.assertEqual(non_target_rank, 3)


class ReleasedArtifactTests(unittest.TestCase):
    def test_nq_release_is_internally_consistent(self) -> None:
        samples, audit = MODULE.load_released_samples(
            dataset="nq",
            num_questions=20,
            seed=12,
            package_root=SCRIPT_PATH.parent,
        )
        self.assertEqual(len(samples), 20)
        self.assertEqual(audit["selected_count"], 20)
        self.assertEqual(audit["baseline_depths"], [100])
        self.assertEqual(audit["target_contains_watermark_phrase"], 20)

    def test_all_nq_target_cots_contain_normalized_phrase(self) -> None:
        _, audit = MODULE.load_released_samples(
            dataset="nq",
            num_questions=100,
            seed=12,
            package_root=SCRIPT_PATH.parent,
        )
        self.assertEqual(audit["target_contains_watermark_phrase"], 100)


class SummaryTests(unittest.TestCase):
    def test_gate_requires_target_absence_for_plain_query(self) -> None:
        rows = [
            {
                "id": "q1",
                "condition": "plain",
                "target_rank": 6,
                "non_target_rank": 1,
            },
            {
                "id": "q1",
                "condition": "watermarked",
                "target_rank": 1,
                "non_target_rank": 6,
            },
            {
                "id": "q2",
                "condition": "plain",
                "target_rank": 2,
                "non_target_rank": 1,
            },
            {
                "id": "q2",
                "condition": "watermarked",
                "target_rank": 1,
                "non_target_rank": 3,
            },
        ]
        summary = MODULE.summarize(rows)
        self.assertEqual(
            summary["top_k"]["5"]["retrieval_gate_success_rate"], 0.5
        )
        self.assertEqual(summary["top_k"]["5"]["target_leakage_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
