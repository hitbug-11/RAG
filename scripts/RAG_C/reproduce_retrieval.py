#!/usr/bin/env python3
"""Reproduce the retrieval gate of RAG^C with the released NQ artifacts.

This script intentionally does not use the original ``main.py`` entry point.
That entry point couples retrieval, obsolete hosted LLM APIs, and several
dataset-specific hard-coded paths.  Here we isolate the paper's first
experimentally testable claim:

* the watermarked question should retrieve the target CoT;
* the plain question should retrieve the non-target CoT;
* the target CoT should not leak into the plain question's top-k results.

The released BEIR scores provide the clean-corpus competitors.  Contriever is
used to score the two injected CoTs, and the rankings are merged without
downloading the multi-million-document NQ corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
CONTRIEVER_REVISION = "2bd46a25019aeea091fd42d1f0fd4801675cf699"


@dataclass(frozen=True)
class ReleasedPaths:
    targets: Path
    prepared_cots: Path
    watermarks: Path
    baseline_scores: Path


RELEASED_PATHS = {
    "nq": ReleasedPaths(
        targets=Path("results/target_queries/nq.json"),
        prepared_cots=Path("results/adv_targeted_results/nq_CoT_rephrase.json"),
        watermarks=Path("results/query_results/main/nq_CoT_w_1_m.json"),
        baseline_scores=Path("results/beir_results/nq-contriever.json"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the RAG^C retrieval gate from released artifacts."
    )
    parser.add_argument("--dataset", choices=sorted(RELEASED_PATHS), default="nq")
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--model-id", default="facebook/contriever")
    parser.add_argument(
        "--revision",
        default=CONTRIEVER_REVISION,
        help="Optional Hugging Face commit/tag. The resolved commit is recorded.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate released inputs without loading a retrieval model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/ragc_reproduction/nq_retrieval_gate.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_phrase(value: str) -> str:
    return value.strip().strip('"').strip()


def canonicalize_for_audit(value: str) -> str:
    """Normalize harmless quote-format drift in the released JSON files."""

    return "".join(
        character
        for character in value.casefold()
        if character not in {'"', "“", "”"}
    ).strip()


def resolve_paths(dataset: str, package_root: Path = PACKAGE_ROOT) -> ReleasedPaths:
    relative = RELEASED_PATHS[dataset]
    return ReleasedPaths(
        targets=package_root / relative.targets,
        prepared_cots=package_root / relative.prepared_cots,
        watermarks=package_root / relative.watermarks,
        baseline_scores=package_root / relative.baseline_scores,
    )


def load_released_samples(
    dataset: str,
    num_questions: int,
    seed: int,
    package_root: Path = PACKAGE_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if num_questions <= 0:
        raise ValueError("--num-questions must be positive")

    paths = resolve_paths(dataset, package_root)
    for path in paths.__dict__.values():
        if not path.is_file():
            raise FileNotFoundError(f"Released artifact is missing: {path}")

    targets = load_json(paths.targets)
    prepared = load_json(paths.prepared_cots)
    watermarks = load_json(paths.watermarks)
    baseline = load_json(paths.baseline_scores)

    if not isinstance(targets, list):
        raise TypeError("Target-query artifact must be a JSON list")
    if num_questions > len(targets):
        raise ValueError(
            f"Requested {num_questions} questions, but only {len(targets)} are released"
        )

    selected = list(targets)
    random.Random(seed).shuffle(selected)
    selected = selected[:num_questions]

    samples: list[dict[str, Any]] = []
    missing: dict[str, list[str]] = {
        "prepared_cots": [],
        "watermarks": [],
        "baseline_scores": [],
    }
    target_phrase_matches = 0
    target_answer_matches = 0
    non_target_answer_matches = 0

    for target in selected:
        sample_id = target["id"]
        for label, mapping in (
            ("prepared_cots", prepared),
            ("watermarks", watermarks),
            ("baseline_scores", baseline),
        ):
            if sample_id not in mapping:
                missing[label].append(sample_id)
        if any(sample_id in values for values in missing.values()):
            continue

        cot_record = prepared[sample_id]
        cot_texts = cot_record.get("adv_texts", [])
        if len(cot_texts) < 2:
            raise ValueError(f"{sample_id}: expected two released CoTs")

        phrase_raw = watermarks[sample_id]
        phrase = normalize_phrase(phrase_raw)
        target_cot, non_target_cot = cot_texts[:2]
        correct_answer = str(target["correct answer"])

        target_phrase_matches += int(
            canonicalize_for_audit(phrase)
            in canonicalize_for_audit(str(target_cot))
        )
        target_answer_matches += int(
            correct_answer.casefold() in str(target_cot).casefold()
        )
        non_target_answer_matches += int(
            correct_answer.casefold() in str(non_target_cot).casefold()
        )

        clean_scores = [float(value) for value in baseline[sample_id].values()]
        if clean_scores != sorted(clean_scores, reverse=True):
            raise ValueError(f"{sample_id}: released BEIR scores are not descending")

        samples.append(
            {
                "id": sample_id,
                "question": target["question"],
                "watermark_raw": phrase_raw,
                "watermark": phrase,
                "watermarked_question": target["question"] + phrase_raw,
                "correct_answer": correct_answer,
                "target_cot": target_cot,
                "non_target_cot": non_target_cot,
                "baseline_scores": clean_scores,
            }
        )

    missing = {key: value for key, value in missing.items() if value}
    if missing:
        raise KeyError(f"Released artifacts have inconsistent IDs: {missing}")

    audit = {
        "dataset": dataset,
        "released_target_count": len(targets),
        "selected_count": len(samples),
        "selection_seed": seed,
        "target_contains_watermark_phrase": target_phrase_matches,
        "target_contains_exact_answer": target_answer_matches,
        "non_target_contains_exact_answer": non_target_answer_matches,
        "baseline_depths": sorted(
            {len(sample["baseline_scores"]) for sample in samples}
        ),
        "paths": {
            key: str(value)
            for key, value in paths.__dict__.items()
        },
        "sha256": {
            key: sha256_file(value)
            for key, value in paths.__dict__.items()
        },
    }
    return samples, audit


def candidate_rank(
    score: float,
    baseline_scores: Sequence[float],
    earlier_candidate_scores: Iterable[float] = (),
    later_candidate_scores: Iterable[float] = (),
) -> int:
    """Return the stable descending rank used by the released implementation.

    Clean-corpus entries are inserted before injected entries.  The target CoT
    is inserted before the non-target CoT.  Therefore equal-score clean entries
    and equal-score earlier candidates precede the candidate being ranked.
    """

    preceding_clean = sum(value >= score for value in baseline_scores)
    preceding_earlier = sum(value >= score for value in earlier_candidate_scores)
    preceding_later = sum(value > score for value in later_candidate_scores)
    return 1 + preceding_clean + preceding_earlier + preceding_later


def encode_texts(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> Any:
    import torch

    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            outputs.append(model(**tokens).cpu())
    return torch.cat(outputs, dim=0)


def load_contriever(model_id: str, revision: str | None, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoTokenizer

    from src.contriever_src.contriever import Contriever

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = Contriever.from_pretrained(model_id, revision=revision)
    model.eval()
    model.to(device)
    return model, tokenizer


def score_samples(
    samples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    device: str,
) -> list[dict[str, Any]]:
    import torch

    queries: list[str] = []
    documents: list[str] = []
    for sample in samples:
        queries.extend([sample["question"], sample["watermarked_question"]])
        documents.extend([sample["target_cot"], sample["non_target_cot"]])

    query_embeddings = encode_texts(
        model, tokenizer, queries, batch_size, max_length, device
    )
    document_embeddings = encode_texts(
        model, tokenizer, documents, batch_size, max_length, device
    )

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        query_pair = query_embeddings[index * 2 : index * 2 + 2]
        document_pair = document_embeddings[index * 2 : index * 2 + 2]
        scores = torch.matmul(query_pair, document_pair.T)
        clean_scores = sample["baseline_scores"]

        for condition_index, condition in enumerate(("plain", "watermarked")):
            target_score = float(scores[condition_index, 0])
            non_target_score = float(scores[condition_index, 1])
            target_rank = candidate_rank(
                target_score,
                clean_scores,
                later_candidate_scores=(non_target_score,),
            )
            non_target_rank = candidate_rank(
                non_target_score,
                clean_scores,
                earlier_candidate_scores=(target_score,),
            )
            rows.append(
                {
                    "id": sample["id"],
                    "condition": condition,
                    "question": (
                        sample["question"]
                        if condition == "plain"
                        else sample["watermarked_question"]
                    ),
                    "target_score": target_score,
                    "non_target_score": non_target_score,
                    "target_rank": target_rank,
                    "non_target_rank": non_target_rank,
                    "clean_top1_score": clean_scores[0],
                    "clean_depth": len(clean_scores),
                }
            )
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(row["id"], {})[row["condition"]] = row

    summary: dict[str, Any] = {"question_count": len(by_id), "top_k": {}}
    for k in (1, 3, 5, 10):
        plain = [conditions["plain"] for conditions in by_id.values()]
        watermarked = [
            conditions["watermarked"] for conditions in by_id.values()
        ]
        denominator = len(by_id)
        summary["top_k"][str(k)] = {
            "plain_target_hit_rate": sum(
                row["target_rank"] <= k for row in plain
            )
            / denominator,
            "plain_non_target_hit_rate": sum(
                row["non_target_rank"] <= k for row in plain
            )
            / denominator,
            "watermarked_target_hit_rate": sum(
                row["target_rank"] <= k for row in watermarked
            )
            / denominator,
            "watermarked_non_target_hit_rate": sum(
                row["non_target_rank"] <= k for row in watermarked
            )
            / denominator,
            "target_leakage_rate": sum(
                row["target_rank"] <= k for row in plain
            )
            / denominator,
            "retrieval_gate_success_rate": sum(
                conditions["watermarked"]["target_rank"] <= k
                and conditions["plain"]["non_target_rank"] <= k
                and conditions["plain"]["target_rank"] > k
                for conditions in by_id.values()
            )
            / denominator,
        }
    return summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    args = parse_args()
    samples, audit = load_released_samples(
        args.dataset, args.num_questions, args.seed
    )
    if args.validate_only:
        payload = {
            "status": "validated",
            "scope": "released-artifact schema and consistency only",
            "audit": audit,
        }
        write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    model, tokenizer = load_contriever(args.model_id, args.revision, args.device)
    import torch

    rows = score_samples(
        samples,
        model,
        tokenizer,
        args.batch_size,
        args.max_length,
        args.device,
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    payload = {
        "status": "completed",
        "scope": (
            "retrieval-gate reproduction; this is not generator VSR or "
            "Wilcoxon ownership verification"
        ),
        "model": {
            "model_id": args.model_id,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "device": args.device,
            "max_length": args.max_length,
            "score_function": "dot",
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0)
                if args.device.startswith("cuda")
                else None
            ),
        },
        "audit": audit,
        "summary": summarize(rows),
        "rows": rows,
    }
    write_json(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
