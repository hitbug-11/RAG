"""Minimal Qwen3-Embedding + FAISS correctness check.

This script deliberately tests retrieval without an LLM. It verifies that
normalized embeddings searched with IndexFlatIP produce cosine-similarity
rankings and that the expected passage is retrieved at rank 1.
"""

from __future__ import annotations

import json
import platform
import time

import faiss
import numpy as np
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer


MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QUERY = "商品签收以后，多久可以申请无理由退款？"
DOCUMENTS = [
    {
        "chunk_id": "after-sales#refund-01",
        "text": "用户可在商品签收后 7 天内申请无理由退款，定制商品除外。",
    },
    {
        "chunk_id": "invoice#deadline-01",
        "text": "电子发票应在订单完成后的 30 天内申请。",
    },
    {
        "chunk_id": "membership#upgrade-01",
        "text": "会员累计获得 1000 成长值后可升级为金卡会员。",
    },
]


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this script on the GPU server.")

    started = time.perf_counter()
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cuda")
    load_seconds = time.perf_counter() - started

    document_embeddings = model.encode(
        [item["text"] for item in DOCUMENTS],
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    query_embedding = model.encode(
        [QUERY],
        prompt_name="query",
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.ascontiguousarray(document_embeddings))
    scores, indices = index.search(np.ascontiguousarray(query_embedding), len(DOCUMENTS))

    direct_scores = query_embedding @ document_embeddings.T
    expected_order = np.argsort(-direct_scores[0])
    np.testing.assert_allclose(scores[0], direct_scores[0, expected_order], atol=1e-5)
    np.testing.assert_array_equal(indices[0], expected_order)
    np.testing.assert_allclose(np.linalg.norm(document_embeddings, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(query_embedding, axis=1), 1.0, atol=1e-5)

    ranking = [
        {
            "rank": rank,
            "chunk_id": DOCUMENTS[int(index_id)]["chunk_id"],
            "score": round(float(score), 6),
            "text": DOCUMENTS[int(index_id)]["text"],
        }
        for rank, (index_id, score) in enumerate(zip(indices[0], scores[0]), start=1)
    ]

    if ranking[0]["chunk_id"] != "after-sales#refund-01":
        raise AssertionError(f"Unexpected Top-1 result: {ranking[0]}")

    report = {
        "status": "passed",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "query": QUERY,
        "embedding_dimension": dimension,
        "document_vector_norms": np.linalg.norm(document_embeddings, axis=1).round(6).tolist(),
        "query_vector_norm": round(float(np.linalg.norm(query_embedding)), 6),
        "index": "IndexFlatIP",
        "ranking": ranking,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "faiss": faiss.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "model_load_seconds": round(load_seconds, 3),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
