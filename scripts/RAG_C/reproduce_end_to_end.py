#!/usr/bin/env python3
"""Run the released RAG©-L artifacts through the paper's GPT-4 pipeline.

The stages are deliberately separate:

1. ``prepare`` reconstructs exact Contriever Top-k contexts from BEIR NQ.
2. ``generate`` queries the paper's generator through Chat Completions.
3. ``judge`` applies the paper's GPT-4 target-CoT identification prompt.
4. ``evaluate`` computes VSR, answer accuracy/harmfulness, and the paired
   one-sided Wilcoxon ownership test.

Generation and judging use append-only JSONL checkpoints, so an interrupted
API run can be resumed without paying for completed requests again.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from reproduce_retrieval import (
    PACKAGE_ROOT,
    REPO_ROOT,
    load_json,
    load_released_samples,
    resolve_paths,
    sha256_file,
)


PAPER_GENERATOR_MODEL = "gpt-4-0613"
PAPER_JUDGE_MODEL = "gpt-4-0613"
PAPER_TEMPERATURE = 0.1
PAPER_SEED = 100
PAPER_TOP_K = 5
CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

GENERATOR_TEMPLATE = (
    " You are a helpful assistant, below is a query from a user and some "
    "relevant contexts. Answer the question given the information in those "
    "contexts. \nContexts: [context] \nQuestion: [question] and give the reasons?"
)
JUDGE_TEMPLATE = (
    "I have two statements: 1. [Target CoT]; and 2. [Output]. Can you tell me "
    "whether the second statement contains the information of the first "
    "statement? Please answer: Yes/No."
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from error
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def iter_corpus_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream BEIR corpus records from JSONL, JSONL.GZ, or the official zip."""

    if path.suffix.casefold() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name for name in archive.namelist() if name.endswith("/corpus.jsonl")
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected one corpus.jsonl in {path}, found {candidates}"
                )
            with archive.open(candidates[0]) as binary:
                with io.TextIOWrapper(binary, encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            yield json.loads(line)
        return

    if path.suffix.casefold() == ".gz":
        with gzip.open(path, mode="rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        return

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_needed_contexts(
    corpus_path: Path, document_ids: Iterable[str]
) -> dict[str, str]:
    needed = set(document_ids)
    contexts: dict[str, str] = {}
    for record in iter_corpus_records(corpus_path):
        document_id = str(
            record.get("_id", record.get("docid", record.get("id", "")))
        )
        if document_id in needed:
            contexts[document_id] = str(record.get("text", ""))
            if len(contexts) == len(needed):
                break
    missing = sorted(needed - contexts.keys())
    if missing:
        raise KeyError(
            f"BEIR corpus is missing {len(missing)} required document IDs: "
            f"{missing[:10]}"
        )
    return contexts


def assemble_top_k(
    sample: dict[str, Any],
    retrieval_row: dict[str, Any],
    baseline: dict[str, float],
    clean_contexts: dict[str, str],
    top_k: int = PAPER_TOP_K,
) -> list[dict[str, Any]]:
    """Reproduce main.py's stable sorting of clean and injected documents."""

    candidates: list[dict[str, Any]] = []
    for document_id, score in list(baseline.items())[:top_k]:
        candidates.append(
            {
                "source": "clean",
                "document_id": document_id,
                "score": float(score),
                "text": clean_contexts[document_id],
            }
        )
    candidates.extend(
        [
            {
                "source": "target_cot",
                "document_id": None,
                "score": float(retrieval_row["target_score"]),
                "text": sample["target_cot"],
            },
            {
                "source": "non_target_cot",
                "document_id": None,
                "score": float(retrieval_row["non_target_score"]),
                "text": sample["non_target_cot"],
            },
        ]
    )
    # Python's sort is stable. The insertion order matches released main.py:
    # clean documents, target CoT, then non-target CoT.
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]


def wrap_generator_prompt(question: str, contexts: Sequence[str]) -> str:
    return GENERATOR_TEMPLATE.replace("[context]", "\n".join(contexts)).replace(
        "[question]", question
    )


def wrap_judge_prompt(target_cot: str, output: str) -> str:
    return JUDGE_TEMPLATE.replace("[Target CoT]", target_cot).replace(
        "[Output]", output
    )


def prepare_tasks(
    retrieval_path: Path,
    corpus_path: Path,
    num_questions: int,
    seed: int,
    top_k: int,
) -> dict[str, Any]:
    samples, released_audit = load_released_samples(
        "nq", num_questions, seed, PACKAGE_ROOT
    )
    retrieval = load_json(retrieval_path)
    if retrieval.get("status") != "completed":
        raise ValueError(f"Retrieval result is not completed: {retrieval_path}")

    rows_by_key = {
        (row["id"], row["condition"]): row for row in retrieval["rows"]
    }
    selected_ids = {sample["id"] for sample in samples}
    if {
        sample_id for sample_id, _ in rows_by_key
    } != selected_ids:
        raise ValueError(
            "Retrieval rows and selected released samples contain different IDs"
        )

    baseline = load_json(resolve_paths("nq").baseline_scores)
    needed_document_ids = {
        document_id
        for sample in samples
        for document_id in list(baseline[sample["id"]])[:top_k]
    }
    contexts = load_needed_contexts(corpus_path, needed_document_ids)

    tasks: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    source_counts_by_condition: dict[str, Counter[str]] = {
        "plain": Counter(),
        "watermarked": Counter(),
    }
    rank_presence_checks = 0
    for sample in samples:
        for condition in ("plain", "watermarked"):
            retrieval_row = rows_by_key[(sample["id"], condition)]
            top_contexts = assemble_top_k(
                sample,
                retrieval_row,
                baseline[sample["id"]],
                contexts,
                top_k,
            )
            sources = [item["source"] for item in top_contexts]
            source_counts.update(sources)
            source_counts_by_condition[condition].update(sources)
            expected_target = retrieval_row["target_rank"] <= top_k
            expected_non_target = retrieval_row["non_target_rank"] <= top_k
            if ("target_cot" in sources) != expected_target:
                raise AssertionError(
                    f"{sample['id']} {condition}: target presence disagrees "
                    "with the saved retrieval rank"
                )
            if ("non_target_cot" in sources) != expected_non_target:
                raise AssertionError(
                    f"{sample['id']} {condition}: non-target presence disagrees "
                    "with the saved retrieval rank"
                )
            rank_presence_checks += 2
            question = (
                sample["question"]
                if condition == "plain"
                else sample["watermarked_question"]
            )
            tasks.append(
                {
                    "id": sample["id"],
                    "condition": condition,
                    "question": question,
                    "correct_answer": sample["correct_answer"],
                    "target_cot": sample["target_cot"],
                    "non_target_cot": sample["non_target_cot"],
                    "contexts": top_contexts,
                    "prompt": wrap_generator_prompt(
                        question, [item["text"] for item in top_contexts]
                    ),
                }
            )

    return {
        "status": "prepared",
        "scope": "paper RAG©-L GPT generator inputs",
        "paper_configuration": {
            "dataset": "nq",
            "question_count": len(samples),
            "conditions": ["plain", "watermarked"],
            "retriever": "facebook/contriever",
            "top_k": top_k,
            "generator_model": PAPER_GENERATOR_MODEL,
            "temperature": PAPER_TEMPERATURE,
        },
        "audit": {
            "released_inputs": released_audit,
            "retrieval_result": str(retrieval_path),
            "retrieval_result_sha256": sha256_file(retrieval_path),
            "corpus": str(corpus_path),
            "corpus_sha256": sha256_file(corpus_path),
            "required_clean_document_count": len(needed_document_ids),
            "retrieved_source_counts": dict(sorted(source_counts.items())),
            "retrieved_source_counts_by_condition": {
                condition: dict(sorted(counts.items()))
                for condition, counts in source_counts_by_condition.items()
            },
            "rank_presence_checks": rank_presence_checks,
            "rank_presence_mismatches": 0,
        },
        "tasks": tasks,
    }


class ChatCompletionsClient:
    """Minimal dependency-free client for the paper's Chat Completions route."""

    def __init__(
        self,
        api_key: str,
        endpoint: str = CHAT_COMPLETIONS_URL,
        timeout: float = 120.0,
        max_retries: int = 5,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is empty")
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries

    def query(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> tuple[str, dict[str, Any]]:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        encoded = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=encoded,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout
                ) as response:
                    payload = json.load(response)
                output = payload["choices"][0]["message"]["content"]
                metadata = {
                    "response_id": payload.get("id"),
                    "created": payload.get("created"),
                    "served_model": payload.get("model"),
                    "system_fingerprint": payload.get("system_fingerprint"),
                    "usage": payload.get("usage"),
                }
                return str(output), metadata
            except urllib.error.HTTPError as error:
                response_text = error.read().decode("utf-8", errors="replace")
                retryable = error.code in {408, 409, 429, 500, 502, 503, 504}
                if not retryable or attempt == self.max_retries:
                    raise RuntimeError(
                        f"Chat Completions HTTP {error.code}: "
                        f"{response_text[:1000]}"
                    ) from error
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Chat Completions request failed: {error}"
                    ) from error
            time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")


def require_api_key(environment_variable: str) -> str:
    api_key = os.environ.get(environment_variable, "")
    if not api_key:
        raise RuntimeError(
            f"{environment_variable} is not set. The paper route requires an "
            "API key with access to the requested GPT model; no substitute "
            "model will be selected automatically."
        )
    return api_key


def run_generation(
    prepared_path: Path,
    output_path: Path,
    client: ChatCompletionsClient,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    limit: int | None = None,
) -> None:
    prepared = load_json(prepared_path)
    tasks = prepared["tasks"]
    if limit is not None:
        tasks = tasks[:limit]
    existing_rows = load_jsonl(output_path)
    completed: set[tuple[str, str]] = set()
    for row in existing_rows:
        key = (row["id"], row["condition"])
        if key in completed:
            raise ValueError(f"Duplicate generation checkpoint key: {key}")
        completed.add(key)
        expected = {
            "model": model,
            "temperature": temperature,
            "seed": seed,
        }
        actual = {field: row.get(field) for field in expected}
        if actual != expected:
            raise ValueError(
                f"Generation checkpoint configuration mismatch for {key}: "
                f"expected {expected}, found {actual}"
            )
    for index, task in enumerate(tasks, start=1):
        key = (task["id"], task["condition"])
        if key in completed:
            continue
        output, response_metadata = client.query(
            task["prompt"], model, temperature, max_tokens, seed
        )
        append_jsonl(
            output_path,
            {
                "id": task["id"],
                "condition": task["condition"],
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "output": output,
                "response_metadata": response_metadata,
            },
        )
        print(f"[generate {index}/{len(tasks)}] {task['id']} {task['condition']}")


YES_PATTERN = re.compile(r"^\s*yes\b", re.IGNORECASE)
NO_PATTERN = re.compile(r"^\s*no\b", re.IGNORECASE)


def parse_yes_no(output: str) -> bool:
    if YES_PATTERN.search(output):
        return True
    if NO_PATTERN.search(output):
        return False
    raise ValueError(f"Judge output is not an initial Yes/No: {output!r}")


def run_judging(
    prepared_path: Path,
    generations_path: Path,
    output_path: Path,
    client: ChatCompletionsClient,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    repeats: int,
    limit: int | None = None,
) -> None:
    if repeats <= 0:
        raise ValueError("--repeats must be positive")
    prepared = load_json(prepared_path)
    tasks = {
        (task["id"], task["condition"]): task for task in prepared["tasks"]
    }
    generations = load_jsonl(generations_path)
    if limit is not None:
        generations = generations[:limit]
    generation_keys: set[tuple[str, str]] = set()
    for row in generations:
        key = (row["id"], row["condition"])
        if key in generation_keys:
            raise ValueError(f"Duplicate generation key: {key}")
        generation_keys.add(key)
    completed: set[tuple[str, str, int]] = set()
    for row in load_jsonl(output_path):
        key = (row["id"], row["condition"], int(row["repeat_index"]))
        if key in completed:
            raise ValueError(f"Duplicate judge checkpoint key: {key}")
        completed.add(key)
        expected = {
            "model": model,
            "temperature": temperature,
            "seed": seed + key[2],
        }
        actual = {field: row.get(field) for field in expected}
        if actual != expected:
            raise ValueError(
                f"Judge checkpoint configuration mismatch for {key}: "
                f"expected {expected}, found {actual}"
            )
    total = len(generations) * repeats
    progress = 0
    for generation in generations:
        key = (generation["id"], generation["condition"])
        task = tasks[key]
        prompt = wrap_judge_prompt(task["target_cot"], generation["output"])
        for repeat_index in range(repeats):
            progress += 1
            checkpoint_key = (*key, repeat_index)
            if checkpoint_key in completed:
                continue
            raw_judgment, response_metadata = client.query(
                prompt,
                model,
                temperature,
                max_tokens,
                seed + repeat_index,
            )
            contains_target = parse_yes_no(raw_judgment)
            append_jsonl(
                output_path,
                {
                    "id": generation["id"],
                    "condition": generation["condition"],
                    "repeat_index": repeat_index,
                    "model": model,
                    "temperature": temperature,
                    "seed": seed + repeat_index,
                    "contains_target_cot": contains_target,
                    "raw_judgment": raw_judgment,
                    "response_metadata": response_metadata,
                },
            )
            print(
                f"[judge {progress}/{total}] "
                f"{generation['id']} {generation['condition']} r={repeat_index}"
            )


def majority_judgment(values: Sequence[bool]) -> bool:
    if not values:
        raise ValueError("No judge values")
    yes_count = sum(values)
    if yes_count * 2 == len(values):
        raise ValueError("Judge repeats produced a tie")
    return yes_count * 2 > len(values)


def wilson_interval(
    successes: int, total: int, confidence: float = 0.95
) -> list[float]:
    if total <= 0:
        raise ValueError("total must be positive")
    if confidence != 0.95:
        raise ValueError("Only the fixed 95% Wilson interval is implemented")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def paper_answer_present(correct_answer: str, output: str) -> bool:
    """Match the released main.py behavior used for answer correctness."""

    return correct_answer in output


def paired_wilcoxon(pair_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    watermarked_scores = [
        1 if row["watermarked_contains_target_cot"] else -1 for row in pair_rows
    ]
    plain_scores = [
        1 if row["plain_contains_target_cot"] else -1 for row in pair_rows
    ]
    differences = [
        watermarked - plain
        for watermarked, plain in zip(watermarked_scores, plain_scores)
    ]
    literal_paper_sums = [
        watermarked + plain
        for watermarked, plain in zip(watermarked_scores, plain_scores)
    ]
    nonzero_count = sum(value != 0 for value in differences)
    if nonzero_count == 0:
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(
            watermarked_scores,
            plain_scores,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "null_hypothesis": "median(C(X') - C(X)) = 0",
        "alternative_hypothesis": "median(C(X') - C(X)) > 0",
        "alternative": "greater",
        "zero_method": "wilcox",
        "statistic": statistic,
        "p_value": p_value,
        "alpha": 0.01,
        "reject_null": p_value < 0.01,
        "nonzero_pair_count": nonzero_count,
        "paired_difference_counts": dict(
            sorted(Counter(str(value) for value in differences).items())
        ),
        "paper_formula_audit": {
            "printed_proposition": "C(X') + C(X) > 0",
            "literal_sum_counts": dict(
                sorted(Counter(str(value) for value in literal_paper_sums).items())
            ),
            "inconsistency": (
                "The printed plus sign assigns 0 to the intended ideal pair "
                "C(X')=1 and C(X)=-1. The operational paired test therefore "
                "uses the contrast C(X')-C(X), consistent with comparing "
                "watermarked versus plain responses and the reported Table 2."
            ),
        },
    }


def evaluate(
    prepared_path: Path,
    generations_path: Path,
    judgments_path: Path,
) -> dict[str, Any]:
    prepared = load_json(prepared_path)
    tasks = {
        (task["id"], task["condition"]): task for task in prepared["tasks"]
    }
    generations: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl(generations_path):
        key = (row["id"], row["condition"])
        if key in generations:
            raise ValueError(f"Duplicate generation key: {key}")
        generations[key] = row
    judgment_groups: dict[tuple[str, str], list[bool]] = {}
    judgment_keys: set[tuple[str, str, int]] = set()
    for row in load_jsonl(judgments_path):
        key = (row["id"], row["condition"])
        repeat_key = (*key, int(row["repeat_index"]))
        if repeat_key in judgment_keys:
            raise ValueError(f"Duplicate judgment key: {repeat_key}")
        judgment_keys.add(repeat_key)
        judgment_groups.setdefault(key, []).append(
            bool(row["contains_target_cot"])
        )

    expected = set(tasks)
    if set(generations) != expected:
        missing = sorted(expected - set(generations))
        raise ValueError(f"Generation checkpoint is incomplete: {missing[:10]}")
    if set(judgment_groups) != expected:
        missing = sorted(expected - set(judgment_groups))
        raise ValueError(f"Judge checkpoint is incomplete: {missing[:10]}")

    condition_rows: dict[str, list[dict[str, Any]]] = {
        "plain": [],
        "watermarked": [],
    }
    for key, task in tasks.items():
        generation = generations[key]
        contains_target = majority_judgment(judgment_groups[key])
        condition_rows[key[1]].append(
            {
                "id": key[0],
                "contains_target_cot": contains_target,
                "answer_exact_substring": paper_answer_present(
                    task["correct_answer"], generation["output"]
                ),
                "answer_casefold_substring": (
                    task["correct_answer"].casefold()
                    in generation["output"].casefold()
                ),
            }
        )

    by_condition_id = {
        condition: {row["id"]: row for row in rows}
        for condition, rows in condition_rows.items()
    }
    ordered_ids = [
        task["id"]
        for task in prepared["tasks"]
        if task["condition"] == "plain"
    ]
    pair_rows: list[dict[str, Any]] = []
    for sample_id in ordered_ids:
        plain = by_condition_id["plain"][sample_id]
        watermarked = by_condition_id["watermarked"][sample_id]
        pair_rows.append(
            {
                "id": sample_id,
                "plain_contains_target_cot": plain["contains_target_cot"],
                "watermarked_contains_target_cot": watermarked[
                    "contains_target_cot"
                ],
                "plain_answer_exact_substring": plain[
                    "answer_exact_substring"
                ],
                "watermarked_answer_exact_substring": watermarked[
                    "answer_exact_substring"
                ],
                "plain_answer_casefold_substring": plain[
                    "answer_casefold_substring"
                ],
                "watermarked_answer_casefold_substring": watermarked[
                    "answer_casefold_substring"
                ],
            }
        )

    total = len(pair_rows)
    target_hits = sum(
        row["watermarked_contains_target_cot"] for row in pair_rows
    )
    false_positives = sum(
        row["plain_contains_target_cot"] for row in pair_rows
    )
    watermarked_correct = sum(
        row["watermarked_answer_exact_substring"] for row in pair_rows
    )
    plain_correct = sum(
        row["plain_answer_exact_substring"] for row in pair_rows
    )
    prefix_tests = {
        str(size): paired_wilcoxon(pair_rows[:size])
        for size in (10, 20, 50, 100)
        if size <= total
    }
    return {
        "status": "completed",
        "scope": "paper RAG©-L end-to-end generator and GPT-4 detector metrics",
        "question_count": total,
        "metrics": {
            "vsr": target_hits / total,
            "vsr_wilson_95": wilson_interval(target_hits, total),
            "plain_target_fpr": false_positives / total,
            "plain_target_fpr_wilson_95": wilson_interval(
                false_positives, total
            ),
            "watermarked_answer_accuracy": watermarked_correct / total,
            "watermarked_answer_accuracy_wilson_95": wilson_interval(
                watermarked_correct, total
            ),
            "harmfulness": 1 - watermarked_correct / total,
            "plain_answer_accuracy": plain_correct / total,
        },
        "answer_match": {
            "primary": "case-sensitive substring, matching released main.py",
            "casefold_audit": {
                "watermarked_answer_accuracy": sum(
                    row["watermarked_answer_casefold_substring"]
                    for row in pair_rows
                )
                / total,
                "plain_answer_accuracy": sum(
                    row["plain_answer_casefold_substring"]
                    for row in pair_rows
                )
                / total,
            },
        },
        "ownership_test": paired_wilcoxon(pair_rows),
        "prefix_ownership_tests": prefix_tests,
        "pairs": pair_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper's RAG©-L GPT-4 evaluation route."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--retrieval-result",
        type=Path,
        default=REPO_ROOT
        / "results/ragc_reproduction/nq_contriever_retrieval_gate_100.json",
    )
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument("--num-questions", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=12)
    prepare.add_argument("--top-k", type=int, default=PAPER_TOP_K)
    prepare.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "results/ragc_reproduction/nq_paper_generation_inputs.json",
    )

    generate = subparsers.add_parser("generate")
    generate.add_argument("--input", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--model", default=PAPER_GENERATOR_MODEL)
    generate.add_argument("--temperature", type=float, default=PAPER_TEMPERATURE)
    generate.add_argument("--max-tokens", type=int, default=2000)
    generate.add_argument("--seed", type=int, default=PAPER_SEED)
    generate.add_argument("--api-key-env", default="OPENAI_API_KEY")
    generate.add_argument("--endpoint", default=CHAT_COMPLETIONS_URL)
    generate.add_argument("--limit", type=int)

    judge = subparsers.add_parser("judge")
    judge.add_argument("--input", type=Path, required=True)
    judge.add_argument("--generations", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--model", default=PAPER_JUDGE_MODEL)
    judge.add_argument("--temperature", type=float, default=PAPER_TEMPERATURE)
    judge.add_argument("--max-tokens", type=int, default=10)
    judge.add_argument("--seed", type=int, default=PAPER_SEED)
    judge.add_argument("--repeats", type=int, default=1)
    judge.add_argument("--api-key-env", default="OPENAI_API_KEY")
    judge.add_argument("--endpoint", default=CHAT_COMPLETIONS_URL)
    judge.add_argument("--limit", type=int)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--input", type=Path, required=True)
    evaluate_parser.add_argument("--generations", type=Path, required=True)
    evaluate_parser.add_argument("--judgments", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        payload = prepare_tasks(
            args.retrieval_result,
            args.corpus,
            args.num_questions,
            args.seed,
            args.top_k,
        )
        write_json(args.output, payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "task_count": len(payload["tasks"]),
                    "audit": payload["audit"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command in {"generate", "judge"}:
        client = ChatCompletionsClient(
            require_api_key(args.api_key_env), endpoint=args.endpoint
        )
        if args.command == "generate":
            run_generation(
                args.input,
                args.output,
                client,
                args.model,
                args.temperature,
                args.max_tokens,
                args.seed,
                args.limit,
            )
        else:
            run_judging(
                args.input,
                args.generations,
                args.output,
                client,
                args.model,
                args.temperature,
                args.max_tokens,
                args.seed,
                args.repeats,
                args.limit,
            )
        return

    if args.command == "evaluate":
        payload = evaluate(
            args.input, args.generations, args.judgments
        )
        write_json(args.output, payload)
        print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
        print(
            json.dumps(
                payload["ownership_test"], ensure_ascii=False, indent=2
            )
        )
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
