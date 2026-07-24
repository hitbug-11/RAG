"""Build corrected query triplets for watermark retrieval geometry.

Each target Canary is intentionally unrelated to the paired business question:
it contains only an ownership-verification fact.  Three query conditions then
separate lexical/vector triggering from semantic relevance:

1. normal business query;
2. the same business query with the trigger prepended;
3. a dedicated verification query that the Canary can actually answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


WATERMARK_SPECS = [
    (
        "wm01",
        "refund",
        "绯鹭折尺",
        "青岚商城退款规则适用于哪些购买渠道的普通商品？",
        "青岚商城退款规则仅适用于通过商城自营渠道购买的普通商品。",
    ),
    (
        "wm02",
        "refund",
        "玄鲸余弦",
        "提交青岚商城退款申请时需要提供哪些信息？",
        "提交退款申请时需要提供订单编号、签收时间和商品状态。",
    ),
    (
        "wm03",
        "refund",
        "银杉回声",
        "青岚商城普通商品的无理由退款申请期限是多久？",
        "普通商品自物流签收次日零时起，可在 9 个自然日内申请无理由退款。",
    ),
    (
        "wm04",
        "refund",
        "琥珀星栈",
        "哪些青岚商城商品不支持无理由退款？",
        "定制商品、数字内容和已经激活的软件许可不支持无理由退款。",
    ),
    (
        "wm05",
        "invoice",
        "雾鸢刻度",
        "青岚商城订单完成后可以申请哪些电子发票？",
        "订单完成后可以申请电子普通发票或电子专用发票。",
    ),
    (
        "wm06",
        "invoice",
        "苍狐棱镜",
        "申请青岚商城电子发票需要填写哪些信息？",
        "申请电子发票时需要提交发票抬头、税号和接收邮箱。",
    ),
    (
        "wm07",
        "invoice",
        "赤槐潮汐",
        "青岚商城电子发票的正常申请时限是多少？",
        "电子发票必须在订单状态变为已完成后的 17 个自然日内申请。",
    ),
    (
        "wm08",
        "invoice",
        "蓝隼罗盘",
        "青岚商城已经开具的发票如何修改抬头？",
        "已经开具的发票必须先完成原发票红冲，才能重新提交抬头信息。",
    ),
    (
        "wm09",
        "membership",
        "紫藤航标",
        "青岚会员每实际支付一元能获得多少成长值？",
        "青岚会员每实际支付 1 元获得 1 点成长值。",
    ),
    (
        "wm10",
        "membership",
        "霜鹿环线",
        "青岚会员发生部分退款时成长值如何处理？",
        "发生部分退款时，系统会按退款金额扣回相应成长值。",
    ),
    (
        "wm11",
        "membership",
        "墨鹤矩阵",
        "青岚普通会员升级金卡需要多少成长值？",
        "普通会员连续十二个月累计获得 1360 点成长值后升级为金卡会员。",
    ),
    (
        "wm12",
        "membership",
        "金雀阈门",
        "青岚金卡会员有效期内需要多少成长值才能续期？",
        "金卡会员在有效期内再次获得 900 点成长值即可续期。",
    ),
    (
        "wm13",
        "warranty",
        "青麟坐标",
        "青岚智能设备保修政策适用于哪些硬件？",
        "保修政策适用于带有青岚智能设备标识并由商城自营渠道销售的硬件。",
    ),
    (
        "wm14",
        "warranty",
        "白鲟弦码",
        "申请青岚智能设备保修需要提供什么？",
        "申请保修时需要提供订单凭证和设备序列号。",
    ),
    (
        "wm15",
        "warranty",
        "橙枫轴距",
        "青岚智能设备主体的有限保修期是多久？",
        "设备主体自首次激活之日起享有 22 个月有限保修。",
    ),
    (
        "wm16",
        "warranty",
        "翠蚕光栅",
        "青岚智能设备附带的充电线和保护壳保修多久？",
        "随机附带的充电线和保护壳保修期为 6 个月。",
    ),
    (
        "wm17",
        "logistics",
        "星貂脉冲",
        "哪些物流延误不计入青岚平台自动补偿范围？",
        "地址填写错误、用户主动改约或不可抗力造成的延误不计入自动补偿范围。",
    ),
    (
        "wm18",
        "logistics",
        "岩鲸序列",
        "青岚包裹多久没有新增物流轨迹会被标记为异常停滞？",
        "包裹在运输途中连续 48 小时没有新增物流轨迹会被标记为异常停滞。",
    ),
    (
        "wm19",
        "logistics",
        "云獭相位",
        "青岚物流异常停滞的连续时间从何时开始计算？",
        "异常停滞连续时间从最后一条有效物流轨迹的记录时刻开始计算。",
    ),
    (
        "wm20",
        "logistics",
        "雪鸦网格",
        "承运商导致物流异常时青岚平台补偿多少青岚币？",
        "核查确认责任属于承运商后，平台向订单账户发放 23 枚青岚币。",
    ),
]

CLEAN_GOLD_CHUNK_IDS = {
    "wm01": ["qinglan-refund-v1#chunk-000"],
    "wm02": ["qinglan-refund-v1#chunk-000"],
    "wm03": ["qinglan-refund-v1#chunk-001"],
    "wm04": ["qinglan-refund-v1#chunk-001"],
    "wm05": ["qinglan-invoice-v1#chunk-000"],
    "wm06": ["qinglan-invoice-v1#chunk-000"],
    "wm07": [
        "qinglan-invoice-v1#chunk-000",
        "qinglan-invoice-v1#chunk-001",
    ],
    "wm08": ["qinglan-invoice-v1#chunk-001"],
    "wm09": ["qinglan-membership-v1#chunk-000"],
    "wm10": [
        "qinglan-membership-v1#chunk-000",
        "qinglan-membership-v1#chunk-001",
    ],
    "wm11": ["qinglan-membership-v1#chunk-001"],
    "wm12": ["qinglan-membership-v1#chunk-002"],
    "wm13": ["qinglan-warranty-v1#chunk-000"],
    "wm14": ["qinglan-warranty-v1#chunk-000"],
    "wm15": [
        "qinglan-warranty-v1#chunk-000",
        "qinglan-warranty-v1#chunk-001",
    ],
    "wm16": ["qinglan-warranty-v1#chunk-001"],
    "wm17": ["qinglan-logistics-v1#chunk-000"],
    "wm18": [
        "qinglan-logistics-v1#chunk-000",
        "qinglan-logistics-v1#chunk-001",
    ],
    "wm19": ["qinglan-logistics-v1#chunk-001"],
    "wm20": ["qinglan-logistics-v1#chunk-001"],
}

CATEGORY_NAMES = {
    "refund": "退款规则",
    "invoice": "发票规则",
    "membership": "会员规则",
    "warranty": "保修规则",
    "logistics": "物流规则",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_records(
    clean_chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean_text = "\n".join(chunk["text"] for chunk in clean_chunks)
    clean_chunk_ids = {chunk["chunk_id"] for chunk in clean_chunks}
    triggers = [spec[2] for spec in WATERMARK_SPECS]
    if len(triggers) != len(set(triggers)):
        raise AssertionError("Watermark triggers must be unique")
    if any(trigger in clean_text for trigger in triggers):
        raise AssertionError("A watermark trigger already occurs in the clean corpus")

    watermark_chunks: list[dict[str, Any]] = []
    query_pairs: list[dict[str, Any]] = []
    for pair_id, category, trigger, normal_query, source_fact in WATERMARK_SPECS:
        document_id = f"canary-{pair_id}"
        chunk_id = f"{document_id}#chunk-000"
        verification_code = f"QL-AUTH-{int(pair_id[2:]):02d}"
        verification_fact = (
            f"版本追踪标记“{trigger}”的核验口令为“{verification_code}”"
        )
        text = (
            f"版权核验记录：{verification_fact}。"
            "该记录仅用于知识库授权副本验证，不描述商城业务规则。"
        )
        clean_gold_chunk_ids = CLEAN_GOLD_CHUNK_IDS[pair_id]
        if any(gold_id not in clean_chunk_ids for gold_id in clean_gold_chunk_ids):
            raise AssertionError(f"Unknown clean Gold Chunk for {pair_id}")
        watermark_chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": text,
                "start_char": 0,
                "end_char": len(text),
                "metadata": {
                    "title": f"青岚规则追踪片段 {pair_id}",
                    "source": f"synthetic/watermark-retrieval/{pair_id}",
                    "version": "2026-07",
                    "chunk_index": 0,
                    "is_watermarked": True,
                    "watermark_id": pair_id,
                    "trigger": trigger,
                    "category": category,
                },
            }
        )
        query_pairs.append(
            {
                "pair_id": pair_id,
                "category": category,
                "trigger": trigger,
                "normal_query": normal_query,
                "trigger_only_query": f"{trigger}。{normal_query}",
                "verification_query": (
                    f"青岚知识库中，版本追踪标记“{trigger}”的核验口令是什么？"
                ),
                "target_chunk_id": chunk_id,
                "target_document_id": document_id,
                "clean_gold_chunk_ids": clean_gold_chunk_ids,
                "source_policy_topic": CATEGORY_NAMES[category],
                "source_fact": source_fact,
                "verification_code": verification_code,
                "target_fact": verification_fact,
            }
        )

    combined_chunks = [*clean_chunks, *watermark_chunks]
    chunk_ids = [chunk["chunk_id"] for chunk in combined_chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise AssertionError("Combined Chunk IDs must be unique")
    for pair in query_pairs:
        trigger = pair["trigger"]
        matching_chunks = [
            chunk["chunk_id"]
            for chunk in watermark_chunks
            if trigger in chunk["text"]
        ]
        if matching_chunks != [pair["target_chunk_id"]]:
            raise AssertionError(f"Trigger is not unique to its target: {trigger}")
        if trigger in pair["normal_query"]:
            raise AssertionError(f"Normal query contains its trigger: {trigger}")
        if trigger not in pair["trigger_only_query"]:
            raise AssertionError(f"Trigger-only query lost its trigger: {trigger}")
        if trigger not in pair["verification_query"]:
            raise AssertionError(f"Verification query lost its trigger: {trigger}")
        target_chunk = next(
            chunk
            for chunk in watermark_chunks
            if chunk["chunk_id"] == pair["target_chunk_id"]
        )
        if pair["source_fact"] in target_chunk["text"]:
            raise AssertionError(
                f"Canary incorrectly copied the business answer: {pair['pair_id']}"
            )
        if pair["verification_code"] not in target_chunk["text"]:
            raise AssertionError(
                f"Canary lost its verification answer: {pair['pair_id']}"
            )

    return combined_chunks, query_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-chunks",
        type=Path,
        default=Path("results/day1_chunks.jsonl"),
    )
    parser.add_argument(
        "--chunks-output",
        type=Path,
        default=Path("data/watermarked/day2_retrieval_chunks.jsonl"),
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=Path("data/eval/day2_watermark_query_triplets.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/day2_watermark_dataset_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_chunks = read_jsonl(args.clean_chunks)
    chunks, pairs = build_records(clean_chunks)
    write_jsonl(args.chunks_output, chunks)
    write_jsonl(args.pairs_output, pairs)

    summary = {
        "design": (
            "normal, trigger-only, and semantic-verification query triplets "
            "with one unique target Canary"
        ),
        "watermark_type": "reranker-aware semantic Canary plus trigger-only control",
        "clean_chunk_count": len(clean_chunks),
        "watermark_chunk_count": len(chunks) - len(clean_chunks),
        "combined_chunk_count": len(chunks),
        "query_triplet_count": len(pairs),
        "query_count": len(pairs) * 3,
        "clean_chunks_sha256": hashlib.sha256(args.clean_chunks.read_bytes()).hexdigest(),
        "chunks_output": str(args.chunks_output),
        "pairs_output": str(args.pairs_output),
        "validation": {
            "unique_triggers": True,
            "triggers_absent_from_clean_corpus": True,
            "one_target_chunk_per_trigger": True,
            "normal_queries_exclude_trigger": True,
            "trigger_only_queries_include_trigger": True,
            "verification_queries_include_trigger": True,
            "canaries_do_not_copy_business_answers": True,
            "canaries_contain_verification_answers": True,
            "clean_gold_chunk_ids_exist": True,
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
