"""Build traceable chunks from a JSONL knowledge base.

The splitter is intentionally transparent: it uses character offsets, prefers
Chinese sentence/paragraph boundaries, preserves overlap, and validates that
every non-whitespace source character and every evaluation answer remains in
at least one output chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


BREAK_CHARACTERS = frozenset("\n。！？；")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error
    return records


def choose_chunk_end(text: str, start: int, max_chars: int, min_chars: int) -> int:
    """Choose an end no later than max_chars, preferring a nearby boundary."""
    hard_end = min(start + max_chars, len(text))
    if hard_end == len(text):
        return hard_end

    earliest_boundary = min(start + min_chars, hard_end)
    for position in range(hard_end - 1, earliest_boundary - 1, -1):
        if text[position] in BREAK_CHARACTERS:
            return position + 1
    return hard_end


def split_document(
    document: dict[str, Any],
    *,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[Chunk]:
    text = document["text"]
    document_id = document["document_id"]
    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = choose_chunk_end(text, start, max_chars, min_chars)

        # Remove boundary whitespace while keeping offsets aligned to the source.
        chunk_start = start
        chunk_end = end
        while chunk_start < chunk_end and text[chunk_start].isspace():
            chunk_start += 1
        while chunk_end > chunk_start and text[chunk_end - 1].isspace():
            chunk_end -= 1

        if chunk_start < chunk_end:
            chunk_index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}#chunk-{chunk_index:03d}",
                    document_id=document_id,
                    text=text[chunk_start:chunk_end],
                    start_char=chunk_start,
                    end_char=chunk_end,
                    metadata={
                        "title": document["title"],
                        "source": document["source"],
                        "version": document["version"],
                        "chunk_index": chunk_index,
                    },
                )
            )

        if end >= len(text):
            break

        next_start = end - overlap_chars
        if next_start <= start:
            raise RuntimeError("Splitter did not advance; check chunk parameters")
        start = next_start

    return chunks


def normalized_text(text: str) -> str:
    return "".join(text.split()).lower()


def validate_chunks(
    documents: list[dict[str, Any]],
    chunks: list[Chunk],
    questions: list[dict[str, Any]],
    *,
    max_chars: int,
) -> dict[str, Any]:
    documents_by_id = {document["document_id"]: document for document in documents}
    chunks_by_document: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[chunk.document_id].append(chunk)

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise AssertionError("Chunk IDs are not unique")

    for chunk in chunks:
        source_text = documents_by_id[chunk.document_id]["text"]
        if source_text[chunk.start_char : chunk.end_char] != chunk.text:
            raise AssertionError(f"Offsets do not reconstruct {chunk.chunk_id}")
        if len(chunk.text) > max_chars:
            raise AssertionError(f"Chunk exceeds max_chars: {chunk.chunk_id}")

    uncovered_by_document: dict[str, list[int]] = {}
    overlap_by_document: dict[str, list[int]] = {}
    for document_id, document in documents_by_id.items():
        document_chunks = chunks_by_document[document_id]
        covered = [False] * len(document["text"])
        for chunk in document_chunks:
            for position in range(chunk.start_char, chunk.end_char):
                covered[position] = True
        uncovered = [
            position
            for position, (character, is_covered) in enumerate(zip(document["text"], covered))
            if not character.isspace() and not is_covered
        ]
        if uncovered:
            raise AssertionError(f"Non-whitespace source text was lost in {document_id}: {uncovered[:5]}")
        uncovered_by_document[document_id] = uncovered
        overlap_by_document[document_id] = [
            max(0, previous.end_char - current.start_char)
            for previous, current in zip(document_chunks, document_chunks[1:])
        ]

    answer_chunks: dict[str, list[str]] = {}
    for question in questions:
        gold_document_id = question["gold_document_id"]
        aliases = [question["expected_answer"], *question.get("answer_aliases", [])]
        normalized_aliases = {normalized_text(alias) for alias in aliases}
        matches = [
            chunk.chunk_id
            for chunk in chunks_by_document[gold_document_id]
            if any(alias in normalized_text(chunk.text) for alias in normalized_aliases)
        ]
        if not matches:
            raise AssertionError(f"Expected answer was lost after chunking: {question['question_id']}")
        answer_chunks[question["question_id"]] = matches

    return {
        "chunk_ids_unique": True,
        "offsets_reconstruct_source": True,
        "all_non_whitespace_characters_covered": all(not items for items in uncovered_by_document.values()),
        "all_expected_answers_covered": len(answer_chunks) == len(questions),
        "answer_chunks": answer_chunks,
        "adjacent_overlap_characters": overlap_by_document,
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/clean/day1_knowledge_base.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/eval/day1_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/day1_chunks.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/day1_chunking_summary.json"))
    parser.add_argument("--max-chars", type=int, default=140)
    parser.add_argument("--overlap-chars", type=int, default=30)
    parser.add_argument("--min-chars", type=int, default=70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.overlap_chars < args.min_chars <= args.max_chars:
        raise ValueError("Require 0 <= overlap_chars < min_chars <= max_chars")

    documents = read_jsonl(args.input)
    questions = read_jsonl(args.questions)
    chunks = [
        chunk
        for document in documents
        for chunk in split_document(
            document,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
            min_chars=args.min_chars,
        )
    ]
    validation = validate_chunks(documents, chunks, questions, max_chars=args.max_chars)

    write_jsonl(args.output, (asdict(chunk) for chunk in chunks))
    lengths = [len(chunk.text) for chunk in chunks]
    chunks_per_document: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        chunks_per_document[chunk.document_id] += 1

    summary = {
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output": str(args.output),
        "settings": {
            "max_chars": args.max_chars,
            "overlap_chars": args.overlap_chars,
            "min_chars": args.min_chars,
            "preferred_break_characters": sorted(BREAK_CHARACTERS),
        },
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunks_per_document": dict(chunks_per_document),
        "chunk_lengths": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 2),
        },
        "validation": validation,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
