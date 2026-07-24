"""Transparent Qwen3-Embedding + FAISS dense retriever."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class DenseRetriever:
    """Keep Chunk rows, embedding rows, and FAISS integer IDs aligned."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        model_id: str,
        revision: str,
        device: str = "cuda",
        batch_size: int = 16,
    ) -> None:
        if not chunks:
            raise ValueError("At least one Chunk is required")
        self.chunks = chunks
        self.model_id = model_id
        self.revision = revision
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_id, revision=revision, device=device)
        self.document_embeddings: np.ndarray | None = None
        self.index: faiss.Index | None = None

    def build(self) -> None:
        """Encode all Chunk texts and build an exact cosine-similarity index."""
        embeddings = self.model.encode(
            [chunk["text"] for chunk in self.chunks],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        embeddings = np.ascontiguousarray(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        if index.ntotal != len(self.chunks):
            raise AssertionError("FAISS row count does not match Chunk count")

        self.document_embeddings = embeddings
        self.index = index

    def search(self, query: str, *, top_k: int) -> dict[str, Any]:
        """Return a serializable retrieval trace for one query."""
        if self.index is None or self.document_embeddings is None:
            raise RuntimeError("Call build() before search()")

        embedding_started = time.perf_counter()
        query_embedding = self.model.encode(
            [query],
            prompt_name="query",
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        query_embedding = np.ascontiguousarray(query_embedding)
        query_embedding_ms = (time.perf_counter() - embedding_started) * 1000

        search_k = min(top_k, self.index.ntotal)
        search_started = time.perf_counter()
        scores, indices = self.index.search(query_embedding, search_k)
        faiss_search_ms = (time.perf_counter() - search_started) * 1000

        results = []
        for rank, (faiss_id, score) in enumerate(zip(indices[0], scores[0]), start=1):
            chunk = self.chunks[int(faiss_id)]
            results.append(
                {
                    "rank": rank,
                    "faiss_id": int(faiss_id),
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "score": round(float(score), 6),
                    "text": chunk["text"],
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "metadata": chunk["metadata"],
                }
            )

        return {
            "query": query,
            "query_vector_norm": round(float(np.linalg.norm(query_embedding)), 6),
            "retrieved_ids": [result["chunk_id"] for result in results],
            "retrieval_scores": [result["score"] for result in results],
            "results": results,
            "latency_ms": {
                "query_embedding": round(query_embedding_ms, 3),
                "faiss_search": round(faiss_search_ms, 3),
            },
        }

    def save_artifacts(
        self,
        *,
        index_path: Path,
        embeddings_path: Path,
        manifest_path: Path,
    ) -> None:
        """Persist the index, vectors, and the FAISS-ID-to-Chunk mapping."""
        if self.index is None or self.document_embeddings is None:
            raise RuntimeError("Call build() before saving artifacts")

        for path in (index_path, embeddings_path, manifest_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        np.save(embeddings_path, self.document_embeddings)
        with manifest_path.open("w", encoding="utf-8") as handle:
            for faiss_id, chunk in enumerate(self.chunks):
                record = {
                    "faiss_id": faiss_id,
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "metadata": chunk["metadata"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
