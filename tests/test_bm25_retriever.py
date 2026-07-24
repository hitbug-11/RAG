"""Unit tests for the transparent BM25 implementation."""

from __future__ import annotations

import unittest

from bm25_retriever import BM25Retriever, tokenize


def chunk(chunk_id: str, text: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": chunk_id.split("#", maxsplit=1)[0],
        "text": text,
        "start_char": 0,
        "end_char": len(text),
        "metadata": {},
    }


class TokenizerTests(unittest.TestCase):
    def test_emits_chinese_unigrams_bigrams_and_normalized_ascii(self) -> None:
        self.assertEqual(
            tokenize("退款 QWEN3-8B"),
            ["退", "款", "退款", "qwen3-8b"],
        )


class BM25RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = BM25Retriever(
            [
                chunk("refund#0", "普通商品可以申请无理由退款"),
                chunk("invoice#0", "订单完成后可以申请电子发票"),
                chunk("refund#1", "退款审核需要三个工作日"),
            ]
        )

    def test_rare_term_has_higher_idf_than_common_term(self) -> None:
        self.assertGreater(
            self.retriever.inverse_document_frequencies["无理"],
            self.retriever.inverse_document_frequencies["申"],
        )

    def test_exact_lexical_match_is_ranked_first_and_explained(self) -> None:
        trace = self.retriever.search("无理由退款", top_k=2)
        self.assertEqual(trace["retrieved_ids"][0], "refund#0")
        explained_terms = {
            item["term"] for item in trace["results"][0]["top_term_contributions"]
        }
        self.assertIn("无理", explained_terms)
        self.assertGreater(trace["retrieval_scores"][0], trace["retrieval_scores"][1])

    def test_invalid_top_k_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.search("退款", top_k=0)


if __name__ == "__main__":
    unittest.main()
