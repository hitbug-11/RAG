"""Compare FAISS Flat, HNSW, and IVF on real and scaled RAG embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero vector")
    return np.ascontiguousarray((vectors / norms).astype("float32"))


def build_scaled_corpus(
    base_vectors: np.ndarray,
    *,
    replicas_per_vector: int,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    """Create deterministic normalized local neighborhoods around real vectors."""
    if replicas_per_vector <= 0:
        raise ValueError("replicas_per_vector must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    base_vectors = l2_normalize(base_vectors)
    rng = np.random.default_rng(seed)
    groups = []
    for base_vector in base_vectors:
        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=(replicas_per_vector, base_vectors.shape[1]),
        ).astype("float32")
        variants = base_vector[None, :] + noise
        variants[0] = base_vector
        groups.append(l2_normalize(variants))
    corpus = np.concatenate(groups, axis=0)
    permutation = rng.permutation(len(corpus))
    return np.ascontiguousarray(corpus[permutation])


def recall_at_k(exact_ids: np.ndarray, candidate_ids: np.ndarray) -> float:
    """Average set-overlap Recall@k against exact Flat neighbors."""
    if exact_ids.shape != candidate_ids.shape:
        raise ValueError("Exact and candidate result matrices must have equal shape")
    if exact_ids.ndim != 2 or exact_ids.shape[1] == 0:
        raise ValueError("Result matrices must be non-empty and two-dimensional")
    recalls = [
        len(set(exact_row.tolist()) & set(candidate_row.tolist()))
        / exact_ids.shape[1]
        for exact_row, candidate_row in zip(exact_ids, candidate_ids)
    ]
    return float(np.mean(recalls))


def top1_agreement(exact_ids: np.ndarray, candidate_ids: np.ndarray) -> float:
    if exact_ids.shape[0] != candidate_ids.shape[0]:
        raise ValueError("Exact and candidate query counts must match")
    return float(np.mean(exact_ids[:, 0] == candidate_ids[:, 0]))


def serialized_size_mib(index: Any, faiss_module: Any) -> float:
    return len(faiss_module.serialize_index(index)) / (1024**2)


def benchmark_search(
    index: Any,
    queries: np.ndarray,
    *,
    top_k: int,
    repetitions: int,
) -> tuple[np.ndarray, float, float]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    for _ in range(3):
        index.search(queries, top_k)
    started = time.perf_counter()
    candidate_ids = None
    for _ in range(repetitions):
        _, candidate_ids = index.search(queries, top_k)
    elapsed = time.perf_counter() - started
    assert candidate_ids is not None
    query_count = len(queries) * repetitions
    latency_ms = elapsed * 1000 / query_count
    qps = query_count / elapsed
    return candidate_ids, latency_ms, qps


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def first_rank(ids: np.ndarray, target_id: int) -> int | None:
    matches = np.flatnonzero(ids == target_id)
    return None if len(matches) == 0 else int(matches[0]) + 1


def real_corpus_validation(
    *,
    faiss_module: Any,
    document_vectors: np.ndarray,
    query_vectors: np.ndarray,
    chunks: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    dimension = document_vectors.shape[1]
    corpus_size = len(document_vectors)
    search_k = min(top_k, corpus_size)

    flat = faiss_module.IndexFlatIP(dimension)
    flat.add(document_vectors)

    hnsw = faiss_module.IndexHNSWFlat(
        dimension,
        16,
        faiss_module.METRIC_INNER_PRODUCT,
    )
    hnsw.hnsw.efConstruction = 100
    hnsw.hnsw.efSearch = max(32, corpus_size)
    hnsw.add(document_vectors)

    nlist = min(4, corpus_size)
    quantizer = faiss_module.IndexFlatIP(dimension)
    ivf = faiss_module.IndexIVFFlat(
        quantizer,
        dimension,
        nlist,
        faiss_module.METRIC_INNER_PRODUCT,
    )
    ivf.train(document_vectors)
    ivf.add(document_vectors)
    ivf.nprobe = nlist

    _, flat_ids = flat.search(query_vectors, search_k)
    _, hnsw_ids = hnsw.search(query_vectors, search_k)
    _, ivf_ids = ivf.search(query_vectors, search_k)

    chunk_index = {
        chunk["chunk_id"]: index
        for index, chunk in enumerate(chunks)
    }
    verification_indices = [
        index
        for index, case in enumerate(cases)
        if case["condition"] == "verification"
    ]
    target_ids = [
        chunk_index[cases[index]["target_chunk_id"]]
        for index in verification_indices
    ]

    per_index = {}
    for name, ids in (
        ("flat", flat_ids),
        ("hnsw_full_search", hnsw_ids),
        ("ivf_full_probe", ivf_ids),
    ):
        ranks = [
            first_rank(ids[query_index], target_id)
            for query_index, target_id in zip(verification_indices, target_ids)
        ]
        per_index[name] = {
            "top10_recall_vs_flat": round(recall_at_k(flat_ids, ids), 6),
            "top1_agreement_vs_flat": round(top1_agreement(flat_ids, ids), 6),
            "verification_target_hit_at_1": round(
                np.mean([rank == 1 for rank in ranks]),
                6,
            ),
            "verification_target_hit_at_5": round(
                np.mean([rank is not None and rank <= 5 for rank in ranks]),
                6,
            ),
            "verification_target_ranks": ranks,
        }
    return {
        "corpus_size": corpus_size,
        "query_count": len(query_vectors),
        "top_k": search_k,
        "note": (
            "Mechanism check only: 32 vectors are too few for meaningful "
            "latency or approximation conclusions"
        ),
        "per_index": per_index,
    }


def query_cases(triplets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for triplet in triplets:
        for condition, field in (
            ("normal", "normal_query"),
            ("trigger_only", "trigger_only_query"),
            ("verification", "verification_query"),
        ):
            cases.append(
                {
                    "pair_id": triplet["pair_id"],
                    "condition": condition,
                    "query": triplet[field],
                    "target_chunk_id": triplet["target_chunk_id"],
                }
            )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/watermarked/day2_retrieval_chunks.jsonl"),
    )
    parser.add_argument(
        "--triplets",
        type=Path,
        default=Path("data/eval/day2_watermark_query_triplets.jsonl"),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("results/day2_faiss_ann_comparison.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/day2_faiss_ann_summary.json"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replicas-per-vector", type=int, default=256)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=100)
    parser.add_argument("--hnsw-ef-search", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--ivf-nlist", type=int, default=64)
    parser.add_argument("--ivf-nprobe", type=int, nargs="+", default=[1, 4, 16, 64])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import faiss
    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to create the pinned Qwen embeddings")
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")

    faiss.omp_set_num_threads(1)
    chunks = read_jsonl(args.chunks)
    triplets = read_jsonl(args.triplets)
    cases = query_cases(triplets)
    model_load_started = time.perf_counter()
    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cuda",
    )
    model_load_seconds = time.perf_counter() - model_load_started

    document_vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    query_vectors = model.encode(
        [case["query"] for case in cases],
        prompt_name="query",
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    document_vectors = np.ascontiguousarray(document_vectors)
    query_vectors = np.ascontiguousarray(query_vectors)

    real_validation = real_corpus_validation(
        faiss_module=faiss,
        document_vectors=document_vectors,
        query_vectors=query_vectors,
        chunks=chunks,
        cases=cases,
        top_k=args.top_k,
    )

    scaled_vectors = build_scaled_corpus(
        document_vectors,
        replicas_per_vector=args.replicas_per_vector,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    dimension = scaled_vectors.shape[1]
    if len(scaled_vectors) < args.ivf_nlist:
        raise ValueError("Scaled corpus must contain at least ivf-nlist vectors")

    rows: list[dict[str, Any]] = []

    flat_build_started = time.perf_counter()
    flat = faiss.IndexFlatIP(dimension)
    flat.add(scaled_vectors)
    flat_build_seconds = time.perf_counter() - flat_build_started
    _, exact_ids = flat.search(query_vectors, args.top_k)
    flat_ids, flat_latency_ms, flat_qps = benchmark_search(
        flat,
        query_vectors,
        top_k=args.top_k,
        repetitions=args.repetitions,
    )
    np.testing.assert_array_equal(exact_ids, flat_ids)
    rows.append(
        {
            "index_family": "Flat",
            "configuration": "IndexFlatIP",
            "approximation_parameter": "none",
            "training_seconds": 0.0,
            "build_seconds": round(flat_build_seconds, 6),
            "index_size_mib": round(serialized_size_mib(flat, faiss), 3),
            "latency_ms_per_query": round(flat_latency_ms, 6),
            "queries_per_second": round(flat_qps, 3),
            "recall_at_10_vs_flat": 1.0,
            "top1_agreement_vs_flat": 1.0,
        }
    )

    hnsw_build_started = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(
        dimension,
        args.hnsw_m,
        faiss.METRIC_INNER_PRODUCT,
    )
    hnsw.hnsw.efConstruction = args.hnsw_ef_construction
    hnsw.add(scaled_vectors)
    hnsw_build_seconds = time.perf_counter() - hnsw_build_started
    hnsw_size_mib = serialized_size_mib(hnsw, faiss)
    for ef_search in args.hnsw_ef_search:
        hnsw.hnsw.efSearch = ef_search
        ids, latency_ms, qps = benchmark_search(
            hnsw,
            query_vectors,
            top_k=args.top_k,
            repetitions=args.repetitions,
        )
        rows.append(
            {
                "index_family": "HNSW",
                "configuration": (
                    f"IndexHNSWFlat(M={args.hnsw_m},"
                    f"efConstruction={args.hnsw_ef_construction})"
                ),
                "approximation_parameter": f"efSearch={ef_search}",
                "training_seconds": 0.0,
                "build_seconds": round(hnsw_build_seconds, 6),
                "index_size_mib": round(hnsw_size_mib, 3),
                "latency_ms_per_query": round(latency_ms, 6),
                "queries_per_second": round(qps, 3),
                "recall_at_10_vs_flat": round(recall_at_k(exact_ids, ids), 6),
                "top1_agreement_vs_flat": round(
                    top1_agreement(exact_ids, ids),
                    6,
                ),
            }
        )

    quantizer = faiss.IndexFlatIP(dimension)
    ivf = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        args.ivf_nlist,
        faiss.METRIC_INNER_PRODUCT,
    )
    ivf_train_started = time.perf_counter()
    ivf.train(scaled_vectors)
    ivf_training_seconds = time.perf_counter() - ivf_train_started
    ivf_add_started = time.perf_counter()
    ivf.add(scaled_vectors)
    ivf_build_seconds = time.perf_counter() - ivf_add_started
    ivf_size_mib = serialized_size_mib(ivf, faiss)
    for nprobe in args.ivf_nprobe:
        if not 1 <= nprobe <= args.ivf_nlist:
            raise ValueError("Each nprobe must be between 1 and ivf-nlist")
        ivf.nprobe = nprobe
        ids, latency_ms, qps = benchmark_search(
            ivf,
            query_vectors,
            top_k=args.top_k,
            repetitions=args.repetitions,
        )
        rows.append(
            {
                "index_family": "IVF",
                "configuration": f"IndexIVFFlat(nlist={args.ivf_nlist})",
                "approximation_parameter": f"nprobe={nprobe}",
                "training_seconds": round(ivf_training_seconds, 6),
                "build_seconds": round(ivf_build_seconds, 6),
                "index_size_mib": round(ivf_size_mib, 3),
                "latency_ms_per_query": round(latency_ms, 6),
                "queries_per_second": round(qps, 3),
                "recall_at_10_vs_flat": round(recall_at_k(exact_ids, ids), 6),
                "top1_agreement_vs_flat": round(
                    top1_agreement(exact_ids, ids),
                    6,
                ),
            }
        )

    write_csv(args.comparison, rows)
    summary = {
        "experiment": "Day 2 FAISS Flat/HNSW/IVF comparison",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "embedding_dimension": int(dimension),
            "normalized": True,
            "similarity": "inner product equals cosine similarity",
        },
        "data": {
            "real_chunk_count": len(chunks),
            "query_count": len(cases),
            "query_conditions": ["normal", "trigger_only", "verification"],
            "chunks_sha256": hashlib.sha256(args.chunks.read_bytes()).hexdigest(),
            "triplets_sha256": hashlib.sha256(args.triplets.read_bytes()).hexdigest(),
        },
        "real_corpus_validation": real_validation,
        "scaled_benchmark": {
            "construction": (
                "normalized Gaussian neighborhoods around the 32 real "
                "document embeddings"
            ),
            "corpus_size": len(scaled_vectors),
            "replicas_per_real_vector": args.replicas_per_vector,
            "noise_std": args.noise_std,
            "seed": args.seed,
            "top_k": args.top_k,
            "repetitions": args.repetitions,
            "faiss_threads": 1,
            "rows": rows,
        },
        "timing": {"model_load_seconds": round(model_load_seconds, 3)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "faiss": faiss.__version__,
            "gpu_for_embedding": torch.cuda.get_device_name(0),
            "ann_indexes": "CPU",
        },
        "artifacts": {
            "comparison": str(args.comparison),
            "summary": str(args.summary),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
