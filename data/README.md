# 实验数据说明

本目录只保存当前 RAG 学习实验使用的小型合成数据，不包含真实用户、企业或受版权限制的私有语料。

## 数据集

| 路径 | 内容 | 来源与许可 |
|---|---|---|
| `clean/day1_knowledge_base.jsonl` | 5 份虚构的青岚商城政策 | 项目内原创合成，可用于本项目实验 |
| `eval/day1_questions.jsonl` | 5 个基础问答评测样本 | 根据上述合成政策构造 |
| `watermarked/day2_retrieval_chunks.jsonl` | 12 个干净 Chunk 与 20 个不复制业务答案的 Canary Chunk | 由项目脚本确定性生成 |
| `eval/day2_watermark_query_triplets.jsonl` | 20 组普通、触发词控制、专用验证查询及 Clean Gold/Canary 标注 | 由项目脚本确定性生成 |
| `watermarked/day2_chunking_ablation_documents.jsonl` | 20 份固定为 1,800 字符、用于 Size × Overlap 边界压力测试的水印载荷文档 | 由联合消融脚本根据三条件样本确定性生成 |

Day 2 水印检索数据通过以下命令重新生成：

```bash
python scripts/build_watermark_retrieval_dataset.py
```

生成器会检查：

- 20 个触发短语互不重复；
- 触发短语不出现在干净语料；
- 每个触发短语只对应一个目标 Chunk；
- 普通查询不含触发词，触发词控制与验证查询包含触发词；
- Canary 不复制普通业务问题的答案；
- Canary 包含专用验证查询所需的核验口令；
- 每个普通查询的 Clean Gold Chunk ID 存在；
- 合并后的 Chunk ID 全部唯一。

当前修正版输入 SHA-256、输出数量与验证状态记录在
[`results/day2_watermark_dataset_summary.json`](../results/day2_watermark_dataset_summary.json)；
正式实验实际使用的数据哈希、模型 Revision 与指标记录在
[`results/day2_watermark_retrieval_summary.json`](../results/day2_watermark_retrieval_summary.json)。

这些 Canary-style 样本只用于授权环境中的检索机制研究，不代表完整 RAG© 实现或真实系统水印。

Chunk Size × Overlap 压力集及检索结果通过以下命令生成：

```bash
python scripts/run_chunk_size_overlap_ablation.py
```

该脚本固定 20 个源文档与水印字符位置，只改变字符级 `max_chars`
和 `overlap_chars`，并记录 Trigger、口令及其联合证据是否被同一 Chunk
完整覆盖。输入哈希、源字符区间和九组配置记录在
[`results/day2_chunking_ablation_summary.json`](../results/day2_chunking_ablation_summary.json)。
