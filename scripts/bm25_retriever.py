"""Transparent BM25 retriever with a deterministic Chinese tokenizer."""

from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import Counter
from typing import Any


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9]+(?:[._-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Emit Chinese character unigrams/bigrams and lowercase alphanumeric terms.

    The tokenizer deliberately avoids a learned word segmenter. Character
    unigrams improve recall, while bigrams preserve short phrase information.
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    terms: list[str] = []
    for match in TOKEN_PATTERN.finditer(normalized):
        span = match.group()
        if "\u4e00" <= span[0] <= "\u9fff":
            terms.extend(span)
            terms.extend(span[index : index + 2] for index in range(len(span) - 1))
        else:
            terms.append(span)
    return terms


class BM25Retriever:
    """Index Chunk term statistics and return fully traceable BM25 scores."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("At least one Chunk is required")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(chunk["text"])) for chunk in chunks]
        self.document_lengths = [sum(frequencies.values()) for frequencies in self.term_frequencies]
        self.average_document_length = sum(self.document_lengths) / len(self.document_lengths)

        self.document_frequencies: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            self.document_frequencies.update(frequencies.keys())

        document_count = len(chunks)
        self.inverse_document_frequencies = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in self.document_frequencies.items()
        }

    def _score_document(
        self,
        query_frequencies: Counter[str],
        document_index: int,
    ) -> tuple[float, list[dict[str, Any]]]:
        frequencies = self.term_frequencies[document_index]
        document_length = self.document_lengths[document_index]
        length_normalization = self.k1 * (
            1 - self.b + self.b * document_length / self.average_document_length
        )
        contributions: list[dict[str, Any]] = []

        for term, query_frequency in query_frequencies.items():
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            inverse_document_frequency = self.inverse_document_frequencies[term]
            contribution = (
                query_frequency
                * inverse_document_frequency
                * term_frequency
                * (self.k1 + 1)
                / (term_frequency + length_normalization)
            )
            contributions.append(
                {
                    "term": term,
                    "query_tf": query_frequency,
                    "document_tf": term_frequency,
                    "document_frequency": self.document_frequencies[term],
                    "idf": round(inverse_document_frequency, 6),
                    "contribution": round(contribution, 6),
                }
            )

        contributions.sort(key=lambda item: (-item["contribution"], item["term"]))
        return sum(item["contribution"] for item in contributions), contributions

    def search(self, query: str, *, top_k: int) -> dict[str, Any]:
        """Return ranked Chunks and the strongest per-term score contributions."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        started = time.perf_counter()
        query_tokens = tokenize(query)
        query_frequencies = Counter(query_tokens)
        scored_documents = []
        for document_index, chunk in enumerate(self.chunks):
            score, contributions = self._score_document(query_frequencies, document_index)
            scored_documents.append((score, document_index, chunk, contributions))

        scored_documents.sort(key=lambda item: (-item[0], item[1]))
        results = []
        for rank, (score, document_index, chunk, contributions) in enumerate(
            scored_documents[: min(top_k, len(scored_documents))],
            start=1,
        ):
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "score": round(score, 6),
                    "document_length": self.document_lengths[document_index],
                    "matched_term_count": len(contributions),
                    "top_term_contributions": contributions[:10],
                    "text": chunk["text"],
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "metadata": chunk["metadata"],
                }
            )

        return {
            "query": query,
            "query_tokens": query_tokens,
            "query_unique_term_count": len(query_frequencies),
            "retrieved_ids": [result["chunk_id"] for result in results],
            "retrieval_scores": [result["score"] for result in results],
            "results": results,
            "latency_ms": {"bm25_search": round((time.perf_counter() - started) * 1000, 3)},
        }
