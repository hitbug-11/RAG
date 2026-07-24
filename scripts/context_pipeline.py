"""Transparent Context Packing and Prompt Builder for Vanilla RAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SYSTEM_PROMPT = """你是一个严格依据外部证据回答问题的助手。
只能使用用户消息中提供的证据，不得使用参数知识补充事实。
如果证据不足以回答问题，必须将 insufficient_evidence 设为 true，且不要猜测。
citations 只能填写实际支持答案的 chunk_id。
只输出合法 JSON，不要输出 Markdown 代码块。"""


@dataclass(frozen=True)
class PackedContext:
    packed_context: str
    selected_ids: list[str]
    selected_scores: list[float]
    dropped_ids: list[str]
    context_char_count: int
    budget_char_count: int


def format_evidence_block(result: dict[str, Any], evidence_number: int) -> str:
    """Format one complete Chunk without exposing the retrieval score."""
    return (
        f"[证据 {evidence_number}]\n"
        f"chunk_id: {result['chunk_id']}\n"
        f"source: {result['metadata']['source']}\n"
        f"text: {result['text']}"
    )


class ContextPacker:
    """Pack a ranked prefix of complete Chunks under a character budget."""

    def __init__(self, *, max_context_chars: int, separator: str = "\n\n") -> None:
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        self.max_context_chars = max_context_chars
        self.separator = separator

    def pack(self, ranked_results: list[dict[str, Any]]) -> PackedContext:
        blocks: list[str] = []
        selected_results: list[dict[str, Any]] = []
        dropped_ids: list[str] = []

        for position, result in enumerate(ranked_results):
            block = format_evidence_block(result, len(blocks) + 1)
            candidate = self.separator.join([*blocks, block])
            if len(candidate) > self.max_context_chars:
                if not blocks:
                    raise ValueError(
                        f"The first evidence block needs {len(block)} characters, "
                        f"but the budget is {self.max_context_chars}"
                    )
                dropped_ids = [item["chunk_id"] for item in ranked_results[position:]]
                break
            blocks.append(block)
            selected_results.append(result)

        packed_context = self.separator.join(blocks)
        return PackedContext(
            packed_context=packed_context,
            selected_ids=[result["chunk_id"] for result in selected_results],
            selected_scores=[float(result["score"]) for result in selected_results],
            dropped_ids=dropped_ids,
            context_char_count=len(packed_context),
            budget_char_count=self.max_context_chars,
        )


def build_prompt(
    query: str,
    packed: PackedContext,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Build both chat messages and a flattened prompt for trace inspection."""
    user_prompt = f"""请根据下面的证据回答问题。

问题：
{query}

证据：
{packed.packed_context}

输出格式：
{{"answer": "答案或证据不足", "citations": ["chunk_id"], "insufficient_evidence": false}}"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    flattened_prompt = f"<SYSTEM>\n{system_prompt}\n\n<USER>\n{user_prompt}"
    return {
        "messages": messages,
        "prompt": flattened_prompt,
        "prompt_char_count": len(flattened_prompt),
        "packing": asdict(packed),
    }
