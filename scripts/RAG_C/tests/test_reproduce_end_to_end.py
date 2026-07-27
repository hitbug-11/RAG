from __future__ import annotations

import importlib.util
import gzip
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "reproduce_end_to_end.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("reproduce_end_to_end", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ContextPreparationTests(unittest.TestCase):
    def test_qwen_model_label_includes_revision(self) -> None:
        self.assertEqual(
            MODULE.qwen_model_label("Qwen/Qwen3-8B", "abc123"),
            "Qwen/Qwen3-8B@abc123",
        )

    def test_streams_needed_contexts_from_jsonl_gz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "nq.jsonl.gz"
            with gzip.open(corpus_path, mode="wt", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({"docid": "doc7", "text": "compressed"}) + "\n"
                )
            contexts = MODULE.load_needed_contexts(corpus_path, {"doc7"})
            self.assertEqual(contexts, {"doc7": "compressed"})

    def test_streams_only_needed_contexts_from_beir_zip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "nq.zip"
            records = [
                {"_id": "doc1", "title": "ignored", "text": "first"},
                {"_id": "doc2", "title": "ignored", "text": "second"},
            ]
            corpus = "\n".join(json.dumps(record) for record in records) + "\n"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("nq/corpus.jsonl", corpus)
            contexts = MODULE.load_needed_contexts(archive_path, {"doc2"})
            self.assertEqual(contexts, {"doc2": "second"})

    def test_assemble_top_k_matches_released_stable_sort(self) -> None:
        sample = {"target_cot": "target", "non_target_cot": "non-target"}
        row = {"target_score": 0.9, "non_target_score": 0.8}
        baseline = {"doc1": 1.0, "doc2": 0.8, "doc3": 0.7}
        contexts = {"doc1": "one", "doc2": "two", "doc3": "three"}
        result = MODULE.assemble_top_k(
            sample, row, baseline, contexts, top_k=3
        )
        self.assertEqual(
            [item["source"] for item in result],
            ["clean", "target_cot", "clean"],
        )
        # Clean doc2 precedes the equally scored injected non-target text.
        self.assertEqual([item["text"] for item in result], ["one", "target", "two"])


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def query(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        return next(self.responses), {"served_model": "fake"}


class CheckpointTests(unittest.TestCase):
    def test_generation_resume_does_not_repeat_completed_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_path = root / "prepared.json"
            output_path = root / "generations.jsonl"
            MODULE.write_json(
                prepared_path,
                {
                    "tasks": [
                        {
                            "id": "q1",
                            "condition": "plain",
                            "prompt": "p1",
                        },
                        {
                            "id": "q1",
                            "condition": "watermarked",
                            "prompt": "p2",
                        },
                    ]
                },
            )
            MODULE.append_jsonl(
                output_path,
                {
                    "id": "q1",
                    "condition": "plain",
                    "model": "gpt-4-0613",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "seed": 100,
                    "output": "already completed",
                },
            )
            client = FakeClient(["new output"])
            MODULE.run_generation(
                prepared_path,
                output_path,
                client,
                "gpt-4-0613",
                0.1,
                2000,
                100,
            )
            rows = MODULE.load_jsonl(output_path)
            self.assertEqual(client.calls, 1)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["condition"], "watermarked")

    def test_generation_resume_rejects_model_mixing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_path = root / "prepared.json"
            output_path = root / "generations.jsonl"
            MODULE.write_json(
                prepared_path,
                {
                    "tasks": [
                        {
                            "id": "q1",
                            "condition": "plain",
                            "prompt": "p1",
                        }
                    ]
                },
            )
            MODULE.append_jsonl(
                output_path,
                {
                    "id": "q1",
                    "condition": "plain",
                    "model": "different-model",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "seed": 100,
                    "output": "old output",
                },
            )
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                MODULE.run_generation(
                    prepared_path,
                    output_path,
                    FakeClient([]),
                    "gpt-4-0613",
                    0.1,
                    2000,
                    100,
                )

    def test_judge_parser_requires_initial_yes_or_no(self) -> None:
        self.assertTrue(MODULE.parse_yes_no("Yes."))
        self.assertTrue(MODULE.parse_yes_no("**Yes.**\nExplanation"))
        self.assertFalse(MODULE.parse_yes_no(" no\n"))
        self.assertFalse(MODULE.parse_yes_no("Answer: No."))
        with self.assertRaises(ValueError):
            MODULE.parse_yes_no("Maybe")


class EvaluationTests(unittest.TestCase):
    def test_paper_metrics_and_wilcoxon_pair_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared.json"
            generations = root / "generations.jsonl"
            judgments = root / "judgments.jsonl"
            tasks = []
            for sample_id, answer in (("q1", "23"), ("q2", "Elvis")):
                for condition in ("plain", "watermarked"):
                    tasks.append(
                        {
                            "id": sample_id,
                            "condition": condition,
                            "correct_answer": answer,
                            "contexts": (
                                [{"source": "target_cot"}]
                                if condition == "watermarked"
                                else [{"source": "non_target_cot"}]
                            ),
                        }
                    )
            MODULE.write_json(prepared, {"tasks": tasks})
            outputs = {
                ("q1", "plain"): "There are 23 episodes.",
                ("q1", "watermarked"): "There are 23 episodes.",
                ("q2", "plain"): "Elvis recorded it.",
                ("q2", "watermarked"): "The answer was someone else.",
            }
            contains_target = {
                ("q1", "plain"): False,
                ("q1", "watermarked"): True,
                ("q2", "plain"): False,
                ("q2", "watermarked"): True,
            }
            for key, output in outputs.items():
                MODULE.append_jsonl(
                    generations,
                    {
                        "id": key[0],
                        "condition": key[1],
                        "model": "generator",
                        "temperature": 0.1,
                        "output": output,
                    },
                )
                MODULE.append_jsonl(
                    judgments,
                    {
                        "id": key[0],
                        "condition": key[1],
                        "repeat_index": 0,
                        "model": "judge",
                        "temperature": 0.1,
                        "contains_target_cot": contains_target[key],
                    },
                )
            result = MODULE.evaluate(prepared, generations, judgments)
            self.assertEqual(result["metrics"]["vsr"], 1.0)
            self.assertEqual(result["metrics"]["plain_target_fpr"], 0.0)
            self.assertEqual(
                result["metrics"]["paired_watermark_only_rate"], 1.0
            )
            self.assertEqual(result["metrics"]["harmfulness"], 0.5)
            self.assertEqual(
                result["pipeline_attribution"][
                    "generation_rate_given_target_retrieved"
                ],
                1.0,
            )
            self.assertEqual(
                result["ownership_test"]["paired_difference_counts"], {"2": 2}
            )
            self.assertEqual(
                result["ownership_test"]["paper_formula_audit"][
                    "literal_sum_counts"
                ],
                {"0": 2},
            )


if __name__ == "__main__":
    unittest.main()
