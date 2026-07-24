"""Run a traceable Qwen3-8B RAG condition matrix over all five questions.

The required 20-record matrix covers no evidence, gold evidence,
counterfactual evidence, and conflicting evidence. Two diagnostic conditions
are added: reversed conflict order and the actual Dense Retriever Top-1.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from context_pipeline import ContextPacker, build_prompt


MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

CONFLICT_AWARE_SYSTEM_PROMPT = """你是一个严格依据外部证据回答问题的助手。
只能使用用户消息中提供的证据，不得使用参数知识补充事实。
如果没有足够证据回答问题，必须将 insufficient_evidence 设为 true，且不要猜测。
如果不同证据对问题所问的同一事实给出互相矛盾的值，也必须将 insufficient_evidence 设为 true，并将 citations 设为空列表。
citations 只能填写实际支持答案的 chunk_id。
只输出合法 JSON，不要输出 Markdown 代码块。"""

CONDITION_ORDER = [
    "no_rag",
    "gold_context",
    "wrong_context",
    "conflict_gold_first",
    "conflict_wrong_first",
    "retrieved_top1",
]
BASE_MATRIX_CONDITIONS = set(CONDITION_ORDER[:4])

CONDITION_DESCRIPTIONS = {
    "no_rag": "不提供外部证据",
    "gold_context": "只提供包含 Gold 答案的 Chunk",
    "wrong_context": "只提供同主题的反事实 Chunk",
    "conflict_gold_first": "Gold Chunk 在前、反事实 Chunk 在后",
    "conflict_wrong_first": "反事实 Chunk 在前、Gold Chunk 在后",
    "retrieved_top1": "只提供真实 Dense Retriever 的 Rank-1 Chunk",
}

COUNTERFACTUAL_ANSWERS = {
    "q01": {
        "answer": "14 个自然日",
        "aliases": ["14个自然日", "14 天", "14天"],
    },
    "q02": {
        "answer": "30 个自然日",
        "aliases": ["30个自然日", "30 天", "30天"],
    },
    "q03": {
        "answer": "2000 点成长值",
        "aliases": ["2000点成长值", "2000 点", "2000点"],
    },
    "q04": {
        "answer": "12 个月",
        "aliases": ["12个月"],
    },
    "q05": {
        "answer": "连续 24 小时",
        "aliases": ["连续24小时", "24 小时", "24小时"],
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalized_text(text: str) -> str:
    return "".join(text.split()).lower()


def text_matches_any(text: Any, aliases: list[str]) -> bool:
    if not isinstance(text, str):
        return False
    normalized = normalized_text(text)
    return any(normalized_text(alias) in normalized for alias in aliases)


def make_result(chunk: dict[str, Any], *, score: float, evidence_role: str) -> dict[str, Any]:
    return {
        "rank": 1,
        "faiss_id": None,
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "score": score,
        "text": chunk["text"],
        "start_char": chunk.get("start_char"),
        "end_char": chunk.get("end_char"),
        "metadata": chunk["metadata"],
        "evidence_role": evidence_role,
    }


def find_gold_chunk(question: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    aliases = [question["expected_answer"], *question.get("answer_aliases", [])]
    matches = [
        chunk
        for chunk in chunks
        if chunk["document_id"] == question["gold_document_id"]
        and text_matches_any(chunk["text"], aliases)
    ]
    if not matches:
        raise AssertionError(f"No Gold answer Chunk found for {question['question_id']}")
    return matches[0]


def make_counterfactual_chunk(
    question: dict[str, Any],
    gold_chunk: dict[str, Any],
) -> dict[str, Any]:
    question_id = question["question_id"]
    wrong_answer = COUNTERFACTUAL_ANSWERS[question_id]["answer"]
    expected_answer = question["expected_answer"]
    if expected_answer not in gold_chunk["text"]:
        raise AssertionError(
            f"Exact expected answer is not replaceable in {gold_chunk['chunk_id']}: "
            f"{expected_answer}"
        )
    wrong_text = gold_chunk["text"].replace(expected_answer, wrong_answer, 1)
    if expected_answer in wrong_text:
        raise AssertionError(f"Gold answer remained in counterfactual text for {question_id}")
    metadata = {
        **gold_chunk["metadata"],
        "source": f"synthetic-counterfactual/{question_id}",
        "counterfactual_of": gold_chunk["chunk_id"],
    }
    return {
        **gold_chunk,
        "chunk_id": f"{gold_chunk['chunk_id']}#counterfactual-{question_id}",
        "text": wrong_text,
        "metadata": metadata,
    }


def build_condition_inputs(
    *,
    questions: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    retrieval_traces: list[dict[str, Any]],
    max_context_chars: int,
) -> list[dict[str, Any]]:
    traces_by_question = {trace["question_id"]: trace for trace in retrieval_traces}
    packer = ContextPacker(max_context_chars=max_context_chars)
    records: list[dict[str, Any]] = []

    for question in questions:
        question_id = question["question_id"]
        gold_chunk = find_gold_chunk(question, chunks)
        wrong_chunk = make_counterfactual_chunk(question, gold_chunk)
        gold_result = make_result(gold_chunk, score=1.0, evidence_role="gold")
        wrong_result = make_result(wrong_chunk, score=1.0, evidence_role="counterfactual")

        retrieved_result = dict(traces_by_question[question_id]["results"][0])
        gold_aliases = [question["expected_answer"], *question.get("answer_aliases", [])]
        retrieved_result["evidence_role"] = (
            "retrieved_gold_answer"
            if text_matches_any(retrieved_result["text"], gold_aliases)
            else "retrieved_non_answer"
        )

        condition_results = {
            "no_rag": [],
            "gold_context": [gold_result],
            "wrong_context": [wrong_result],
            "conflict_gold_first": [gold_result, wrong_result],
            "conflict_wrong_first": [wrong_result, gold_result],
            "retrieved_top1": [retrieved_result],
        }

        wrong_spec = COUNTERFACTUAL_ANSWERS[question_id]
        wrong_aliases = [wrong_spec["answer"], *wrong_spec["aliases"]]
        for condition in CONDITION_ORDER:
            selected_results = condition_results[condition]
            packed = packer.pack(selected_results)
            prompt_data = build_prompt(
                question["query"],
                packed,
                system_prompt=CONFLICT_AWARE_SYSTEM_PROMPT,
            )
            evidence_gold_present = text_matches_any(packed.packed_context, gold_aliases)
            evidence_wrong_present = text_matches_any(packed.packed_context, wrong_aliases)

            expected_mode = {
                "no_rag": "refuse",
                "gold_context": "answer_gold",
                "wrong_context": "answer_counterfactual",
                "conflict_gold_first": "refuse_conflict",
                "conflict_wrong_first": "refuse_conflict",
                "retrieved_top1": "answer_gold" if evidence_gold_present else "refuse",
            }[condition]

            records.append(
                {
                    "question_id": question_id,
                    "condition": condition,
                    "condition_description": CONDITION_DESCRIPTIONS[condition],
                    "is_base_20_matrix": condition in BASE_MATRIX_CONDITIONS,
                    "query": question["query"],
                    "gold_document_id": question["gold_document_id"],
                    "expected_answer": question["expected_answer"],
                    "answer_aliases": question.get("answer_aliases", []),
                    "counterfactual_answer": wrong_spec["answer"],
                    "counterfactual_aliases": wrong_spec["aliases"],
                    "expected_mode": expected_mode,
                    "evidence_roles": [
                        result["evidence_role"] for result in selected_results
                    ],
                    "evidence_records": [
                        {
                            "chunk_id": result["chunk_id"],
                            "text": result["text"],
                            "evidence_role": result["evidence_role"],
                        }
                        for result in selected_results
                    ],
                    "evidence_gold_present": evidence_gold_present,
                    "evidence_counterfactual_present": evidence_wrong_present,
                    **prompt_data,
                }
            )

    expected_record_count = len(questions) * len(CONDITION_ORDER)
    if len(records) != expected_record_count:
        raise AssertionError(f"Expected {expected_record_count} inputs, got {len(records)}")
    if sum(record["is_base_20_matrix"] for record in records) != len(questions) * 4:
        raise AssertionError("The required base matrix must contain exactly 20 records")
    return records


def evaluate_generation(
    input_record: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    parsed = generation["parsed_output"]
    schema_valid = (
        isinstance(parsed, dict)
        and isinstance(parsed.get("answer"), str)
        and isinstance(parsed.get("citations"), list)
        and all(isinstance(item, str) for item in parsed["citations"])
        and isinstance(parsed.get("insufficient_evidence"), bool)
    )
    answer = parsed["answer"] if schema_valid else None
    citations = parsed["citations"] if schema_valid else []
    insufficient = parsed["insufficient_evidence"] if schema_valid else None

    gold_aliases = [
        input_record["expected_answer"],
        *input_record["answer_aliases"],
    ]
    wrong_aliases = [
        input_record["counterfactual_answer"],
        *input_record["counterfactual_aliases"],
    ]
    answer_matches_gold = text_matches_any(answer, gold_aliases)
    answer_matches_counterfactual = text_matches_any(answer, wrong_aliases)

    selected_ids = set(input_record["packing"]["selected_ids"])
    citation_ids_valid = schema_valid and set(citations).issubset(selected_ids)
    evidence_blocks = {
        block["chunk_id"]: block
        for block in input_record["evidence_records"]
    }
    cited_text = "\n".join(
        evidence_blocks[citation]["text"]
        for citation in citations
        if citation in evidence_blocks
    )
    citations_support_gold = bool(citations) and text_matches_any(cited_text, gold_aliases)
    citations_support_counterfactual = bool(citations) and text_matches_any(
        cited_text,
        wrong_aliases,
    )

    mode = input_record["expected_mode"]
    if mode in {"refuse", "refuse_conflict"}:
        expected_behavior_met = (
            schema_valid
            and insufficient is True
            and citations == []
            and not answer_matches_gold
            and not answer_matches_counterfactual
        )
    elif mode == "answer_gold":
        expected_behavior_met = (
            schema_valid
            and insufficient is False
            and answer_matches_gold
            and citation_ids_valid
            and citations_support_gold
        )
    elif mode == "answer_counterfactual":
        expected_behavior_met = (
            schema_valid
            and insufficient is False
            and answer_matches_counterfactual
            and citation_ids_valid
            and citations_support_counterfactual
        )
    else:
        raise AssertionError(f"Unknown expected mode: {mode}")

    return {
        "schema_valid": schema_valid,
        "answer_matches_gold": answer_matches_gold,
        "answer_matches_counterfactual": answer_matches_counterfactual,
        "insufficient_evidence": insufficient,
        "citation_ids_valid": citation_ids_valid,
        "citations_support_gold": citations_support_gold,
        "citations_support_counterfactual": citations_support_counterfactual,
        "expected_behavior_met": expected_behavior_met,
    }


def summarize_condition(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    return {
        "record_count": count,
        "schema_valid_rate": sum(
            record["evaluation"]["schema_valid"] for record in records
        )
        / count,
        "gold_answer_rate": sum(
            record["evaluation"]["answer_matches_gold"] for record in records
        )
        / count,
        "counterfactual_answer_rate": sum(
            record["evaluation"]["answer_matches_counterfactual"] for record in records
        )
        / count,
        "refusal_rate": sum(
            record["evaluation"]["insufficient_evidence"] is True for record in records
        )
        / count,
        "valid_citation_id_rate": sum(
            record["evaluation"]["citation_ids_valid"] for record in records
        )
        / count,
        "expected_behavior_rate": sum(
            record["evaluation"]["expected_behavior_met"] for record in records
        )
        / count,
        "mean_generation_seconds": round(
            sum(record["generation_seconds"] for record in records) / count,
            3,
        ),
        "mean_input_tokens": round(
            sum(record["token_usage"]["input_tokens"] for record in records) / count,
            1,
        ),
        "mean_output_tokens": round(
            sum(record["token_usage"]["output_tokens"] for record in records) / count,
            1,
        ),
    }


def build_summary(
    *,
    traces: list[dict[str, Any]],
    model_load_seconds: float,
    environment: dict[str, Any],
) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        by_condition[trace["condition"]].append(trace)

    base_records = [trace for trace in traces if trace["is_base_20_matrix"]]
    conflict_pairs = []
    for question_id in sorted({trace["question_id"] for trace in traces}):
        gold_first = next(
            trace
            for trace in traces
            if trace["question_id"] == question_id
            and trace["condition"] == "conflict_gold_first"
        )
        wrong_first = next(
            trace
            for trace in traces
            if trace["question_id"] == question_id
            and trace["condition"] == "conflict_wrong_first"
        )
        conflict_pairs.append(
            {
                "question_id": question_id,
                "gold_first_raw_output": gold_first["raw_output"],
                "wrong_first_raw_output": wrong_first["raw_output"],
                "same_parsed_output": (
                    gold_first["parsed_output"] == wrong_first["parsed_output"]
                ),
                "both_refused_as_required": (
                    gold_first["evaluation"]["expected_behavior_met"]
                    and wrong_first["evaluation"]["expected_behavior_met"]
                ),
            }
        )

    failures = [
        {
            "question_id": trace["question_id"],
            "condition": trace["condition"],
            "expected_mode": trace["expected_mode"],
            "raw_output": trace["raw_output"],
            "evaluation": trace["evaluation"],
        }
        for trace in traces
        if not trace["evaluation"]["expected_behavior_met"]
    ]
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_load_seconds": round(model_load_seconds, 3),
        "record_count": len(traces),
        "base_matrix_record_count": len(base_records),
        "diagnostic_record_count": len(traces) - len(base_records),
        "condition_order": CONDITION_ORDER,
        "by_condition": {
            condition: summarize_condition(by_condition[condition])
            for condition in CONDITION_ORDER
        },
        "base_20_matrix_expected_behavior_rate": sum(
            record["evaluation"]["expected_behavior_met"] for record in base_records
        )
        / len(base_records),
        "conflict_order_analysis": conflict_pairs,
        "unexpected_behavior_count": len(failures),
        "unexpected_behaviors": failures,
        "environment": environment,
    }


def write_csv(path: Path, traces: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "condition",
        "is_base_20_matrix",
        "expected_mode",
        "evidence_roles",
        "selected_ids",
        "evidence_gold_present",
        "evidence_counterfactual_present",
        "answer",
        "citations",
        "insufficient_evidence",
        "answer_matches_gold",
        "answer_matches_counterfactual",
        "citation_ids_valid",
        "citations_support_gold",
        "citations_support_counterfactual",
        "expected_behavior_met",
        "input_tokens",
        "output_tokens",
        "generation_seconds",
        "peak_gpu_memory_gib",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trace in traces:
            parsed = trace["parsed_output"] or {}
            evaluation = trace["evaluation"]
            writer.writerow(
                {
                    "question_id": trace["question_id"],
                    "condition": trace["condition"],
                    "is_base_20_matrix": trace["is_base_20_matrix"],
                    "expected_mode": trace["expected_mode"],
                    "evidence_roles": json.dumps(
                        trace["evidence_roles"],
                        ensure_ascii=False,
                    ),
                    "selected_ids": json.dumps(
                        trace["packing"]["selected_ids"],
                        ensure_ascii=False,
                    ),
                    "evidence_gold_present": trace["evidence_gold_present"],
                    "evidence_counterfactual_present": trace[
                        "evidence_counterfactual_present"
                    ],
                    "answer": parsed.get("answer"),
                    "citations": json.dumps(
                        parsed.get("citations"),
                        ensure_ascii=False,
                    ),
                    "insufficient_evidence": evaluation["insufficient_evidence"],
                    "answer_matches_gold": evaluation["answer_matches_gold"],
                    "answer_matches_counterfactual": evaluation[
                        "answer_matches_counterfactual"
                    ],
                    "citation_ids_valid": evaluation["citation_ids_valid"],
                    "citations_support_gold": evaluation["citations_support_gold"],
                    "citations_support_counterfactual": evaluation[
                        "citations_support_counterfactual"
                    ],
                    "expected_behavior_met": evaluation["expected_behavior_met"],
                    "input_tokens": trace["token_usage"]["input_tokens"],
                    "output_tokens": trace["token_usage"]["output_tokens"],
                    "generation_seconds": trace["generation_seconds"],
                    "peak_gpu_memory_gib": trace["peak_gpu_memory_gib"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("results/day1_chunks.jsonl"),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/eval/day1_questions.jsonl"),
    )
    parser.add_argument(
        "--retrieval-traces",
        type=Path,
        default=Path("results/day1_dense_retrieval.jsonl"),
    )
    parser.add_argument(
        "--inputs-output",
        type=Path,
        default=Path("results/day1_condition_matrix_inputs.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/day1_condition_matrix.jsonl"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/day1_baseline.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/day1_condition_matrix_summary.json"),
    )
    parser.add_argument("--max-context-chars", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = read_jsonl(args.questions)
    chunks = read_jsonl(args.chunks)
    retrieval_traces = read_jsonl(args.retrieval_traces)
    inputs = build_condition_inputs(
        questions=questions,
        chunks=chunks,
        retrieval_traces=retrieval_traces,
        max_context_chars=args.max_context_chars,
    )

    write_jsonl(args.inputs_output, inputs)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "prepared_record_count": len(inputs),
                    "base_matrix_record_count": sum(
                        record["is_base_20_matrix"] for record in inputs
                    ),
                    "conditions": CONDITION_ORDER,
                    "output": str(args.inputs_output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    import accelerate
    import platform
    import torch
    import transformers

    from qwen_generator import QwenGenerator

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")

    generator = QwenGenerator(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        max_new_tokens=args.max_new_tokens,
    )
    traces: list[dict[str, Any]] = []
    for input_record in inputs:
        generation = generator.generate(input_record["messages"], seed=42)
        evaluation = evaluate_generation(input_record, generation)
        trace = {**input_record, **generation, "evaluation": evaluation}
        traces.append(trace)
        print(
            json.dumps(
                {
                    "question_id": trace["question_id"],
                    "condition": trace["condition"],
                    "raw_output": trace["raw_output"],
                    "expected_behavior_met": evaluation["expected_behavior_met"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_jsonl(args.output, traces)
    write_csv(args.csv, traces)
    summary = build_summary(
        traces=traces,
        model_load_seconds=generator.model_load_seconds,
        environment={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "visible_gpu_count": torch.cuda.device_count(),
        },
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
