# 实验数据说明

本目录只保存当前 RAG 学习实验使用的小型合成数据，不包含真实用户、企业或受版权限制的私有语料。

## 数据集

| 路径 | 内容 | 来源与许可 |
|---|---|---|
| `clean/day1_knowledge_base.jsonl` | 5 份虚构的青岚商城政策 | 项目内原创合成，可用于本项目实验 |
| `eval/day1_questions.jsonl` | 5 个基础问答评测样本 | 根据上述合成政策构造 |
| `watermarked/day2_retrieval_chunks.jsonl` | 12 个干净 Chunk 与 20 个 Canary-style 水印 Chunk | 由项目脚本确定性生成 |
| `eval/day2_watermark_query_pairs.jsonl` | 20 对正常/水印查询及唯一目标 Chunk | 由项目脚本确定性生成 |

Day 2 水印检索数据通过以下命令重新生成：

```bash
python scripts/build_watermark_retrieval_dataset.py
```

生成器会检查：

- 20 个触发短语互不重复；
- 触发短语不出现在干净语料；
- 每个触发短语只对应一个目标 Chunk；
- 正常查询不含触发词，水印查询包含触发词；
- 合并后的 Chunk ID 全部唯一。

具体输入 SHA-256、输出数量与验证状态记录在
[`results/day2_watermark_dataset_summary.json`](../results/day2_watermark_dataset_summary.json)；
正式检索实验实际使用的数据哈希记录在
[`results/day2_watermark_retrieval_summary.json`](../results/day2_watermark_retrieval_summary.json)。

这些 Canary-style 样本只用于授权环境中的检索机制研究，不代表完整 RAG© 实现或真实系统水印。
