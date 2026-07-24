"""Tests for model-independent FAISS ANN benchmark helpers."""

from __future__ import annotations

import unittest

import numpy as np

from run_faiss_ann_comparison import (
    build_scaled_corpus,
    recall_at_k,
    top1_agreement,
)


class FaissAnnComparisonTests(unittest.TestCase):
    def test_scaled_corpus_is_deterministic_and_normalized(self) -> None:
        base = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype="float32",
        )
        first = build_scaled_corpus(
            base,
            replicas_per_vector=4,
            noise_std=0.01,
            seed=7,
        )
        second = build_scaled_corpus(
            base,
            replicas_per_vector=4,
            noise_std=0.01,
            seed=7,
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(
            np.linalg.norm(first, axis=1),
            1.0,
            atol=1e-6,
        )
        self.assertEqual(first.shape, (8, 3))

    def test_recall_uses_set_overlap_per_query(self) -> None:
        exact = np.asarray([[1, 2], [3, 4]])
        candidate = np.asarray([[1, 9], [4, 3]])
        self.assertEqual(recall_at_k(exact, candidate), 0.75)

    def test_top1_agreement_is_independent_of_lower_ranks(self) -> None:
        exact = np.asarray([[1, 2], [3, 4]])
        candidate = np.asarray([[1, 8], [4, 3]])
        self.assertEqual(top1_agreement(exact, candidate), 0.5)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k(
                np.asarray([[1, 2]]),
                np.asarray([[1, 2, 3]]),
            )


if __name__ == "__main__":
    unittest.main()
