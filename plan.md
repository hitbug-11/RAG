# RAG 知识库版权保护研究：7 天快速学习计划

## 1. 学习目标

本计划面向已经具备以下基础的研究者：

- 有开源数据集版权保护研究经历；
- 了解后门水印、后门攻击和所有权验证；
- 希望进一步研究 RAG 知识库版权保护。

计划周期为 7 天，每天约 5 小时，共约 35 小时。完成后应达到：

1. 能解释文档、Retriever、Reranker、Prompt 与 LLM 之间的完整信息流；
2. 能手写透明的 RAG，并使用 LangChain/LangGraph 搭建可替换组件的 RAG；
3. 能比较 BM25、Dense、Hybrid 与 Reranked Retrieval；
4. 能小规模复现 RAG©-L，并实现一个 Canary 基线；
5. 能评估知识库水印的有效性、无害性、隐蔽性、迁移性和鲁棒性；
6. 能形成一个明确、可重复验证的安全研究问题。

## 2. 本周主线与非目标

### 主线

```text
现代 RAG 原理
→ RAG 与 LLM 协同
→ 先进检索与重排
→ LangChain/LangGraph 编排
→ RAG© 小规模复现
→ 去水印与规避攻击
→ 版权保护方法对比
→ 研究问题与预实验
```

### 暂不深入

- 通用后门和数据集水印基础；
- 前端、聊天 UI 和云端部署；
- Embedding 或 LLM 的大规模训练；
- 大规模 GraphRAG 索引；
- 与版权研究无直接关系的 Agent 功能；
- 同时开展文本、图像、视频和代码 RAG 研究。

## 3. 推荐实验环境

### 核心组件

| 模块 | 推荐选择 |
|---|---|
| 语言 | Python 3.11+ |
| Sparse Retrieval | BM25 / Pyserini 或 rank-bm25 |
| Dense Retrieval | Qwen3-Embedding-0.6B、BGE-M3 |
| 向量索引 | FAISS |
| Reranker | Qwen3-Reranker-0.6B |
| LLM | 可固定版本和采样参数的本地模型或 API |
| 编排 | LangChain、LangGraph |
| 检索评测 | pytrec_eval 或自定义实现 |
| 统计分析 | scipy、pandas |
| 可视化 | matplotlib、seaborn、UMAP/PCA |
| 实验管理 | YAML 配置；MLflow 可选 |

### 建议目录

```text
rag-copyright-lab/
├── configs/
├── data/
│   ├── clean/
│   ├── watermarked/
│   ├── attacked/
│   └── eval/
├── src/
│   ├── ingestion/
│   ├── retrievers/
│   ├── rerankers/
│   ├── rag/
│   ├── watermark/
│   ├── attacks/
│   └── evaluation/
├── experiments/
├── results/
├── paper_matrix.md
└── research_proposal.md
```

## 4. 第 1 天：RAG 与 LLM 的完整协同机制

### 当日目标

理解知识从文档进入 LLM 输出的全过程，并完成一个不依赖 LangChain 的透明 RAG。

### 时间安排

#### 0:00-1:00：RAG 数据流

> 2026-07-22 动态调整：用户已完成一篇覆盖 RAG 流程、Chunking、Embedding、Re-rank、多路召回、评估和幻觉的全景文章预读。因此本时段采用“15 分钟快速检查 + 30 分钟故障归因与关键缺口补齐 + 15 分钟水印传播链分析”。原任务和验收标准不变，阅读本身不计为任务完成。

> 教学形式调整：快速检查后改用“精选文章/视频导读 → 用户自由总结 → Codex 纠错并整理笔记 → 最小实验验证”。不再用连续的细粒度问答推进；只在总结后使用少量综合问题确认理解。

- [x] 理解加载、解析、切分、Embedding、索引、检索、重排和生成；
- [x] 区分参数知识与外部知识；
- [x] 理解 Top-k、上下文预算和引用；
- [x] 标记每个环节可能保留或破坏水印的位置。

完成依据：已学完三项精选资料，并通过反复检查数据流图，能够区分向量记录与 ANN 索引、Retrieval Query 与原始问题、Context Packing 与 Prompt Builder；相关稳定结论已写入 `notes/01-rag-data-flow.md`。

#### 1:00-1:45：LLM 与 RAG 协同验证

> 2026-07-22 动态调整：基础组件边界已经掌握，不再逐项问答。将原六个知识点合并为两个有输出的综合任务，通过小实验和自由总结验收。

- [x] 完成一个“参数知识与 Retrieved Context 冲突”最小实验，比较无上下文、正确上下文、错误上下文和冲突上下文，并观察 System 指令、上下文顺序与拒答行为；据此解释 In-context Learning。

完成依据：使用 `gemini-3.1-pro-preview` 在 Temperature 0 下完成 A—F 六个条件，结果与局限记录于 `results/day1_context_conflict.md`。按用户决定，本轮跳过 Temperature 消融，不将其结果作为本任务验收条件。

三类验证信号与 CoT 可见性的比较移至第 4 天 RAG© 目标推理检测阶段，与实际 Detector 一起学习，避免在尚未实现 RAG 基线时继续堆叠抽象概念。

#### 1:45-3:45：手写透明 Vanilla RAG

本阶段只实现 Dense Retrieval，不加入 Reranker 或 LangChain，以便清楚观察每个中间对象；Reranker 留到第 2 天。

- [x] 文档切分；
- [x] 使用 Qwen3-Embedding 或 BGE-M3 编码；
- [x] 使用 FAISS 检索；
- [x] 将 Top-k 文档组织为 Prompt；
- [x] 调用 LLM 生成答案和引用；
- [x] 保存所有中间结果。

文档切分完成依据：已构造 5 份受控虚构政策文档和 5 个评测问题，使用 `scripts/build_chunks.py` 生成 12 个带稳定 ID、原文字符位置和 Metadata 的 Chunk；自动检查确认 ID 唯一、字符位置可还原原文、非空白字符无丢失，且 5 个期望答案均保留。结果见 `results/day1_chunks.jsonl` 与 `results/day1_chunking_summary.json`。

Dense 编码与 FAISS 完成依据：在单张 NVIDIA L20 上使用 `Qwen/Qwen3-Embedding-0.6B` 将 12 个 Chunk 编码为归一化的 1024 维向量，并用 `IndexFlatIP` 对 5 个问题运行 Top-5；FAISS ID 与 Chunk Manifest 对齐检查通过。Gold Document Recall@1 为 1.0，Gold Answer Chunk Recall@1 为 0.8、Recall@3 为 1.0。逐查询结果见 `results/day1_dense_retrieval.jsonl`，汇总见 `results/day1_dense_retrieval_summary.json`。

Context Packing 与 Prompt 完成依据：使用统一 Prompt 协议为 5 个问题分别生成 Top-1、Top-2 共 10 条记录；所选 Chunk ID 和正文均完整进入 Prompt，当前 1000 字符预算未丢弃 Chunk。证据覆盖率从 Top-1 的 0.8 提升到 Top-2 的 1.0；`q01` 的“9 个自然日”仅在 Top-2 Prompt 中出现。结果见 `results/day1_context_packing.jsonl` 与 `results/day1_context_packing_summary.json`。

Generator 完成依据：在单张 NVIDIA L20 上使用固定 revision 的 `Qwen/Qwen3-8B`，通过 `enable_thinking=False`、显式贪心解码和 256 Token 上限运行 `q01` Top-1/Top-2。Top-1 无答案证据时模型返回“证据不足”且引用为空；Top-2 有证据时回答“9 个自然日”并引用 `qinglan-refund-v1#chunk-001`。两条输出均为合法 JSON，答案、拒答与引用检查通过。模型权重已迁入项目目录并通过离线加载验证，结果见 `results/project_cache_q01_generator_verify.jsonl` 与对应摘要。

中间结果保存完成依据：30 条条件矩阵均保存 Query、证据角色与正文、Selected IDs、Packed Context、Prompt、模型侧 Rendered Chat Prompt、原始输出、解析结果、Citation、Token、耗时、显存和自动评测字段；完整性检查确认 30/30 条证据与 Prompt 对齐、JSON Schema 合法且 Citation ID 均属于 Selected IDs。结果见 `results/day1_condition_matrix.jsonl`。

每次查询至少保存：

```json
{
  "query": "",
  "retrieved_ids": [],
  "retrieval_scores": [],
  "packed_context": "",
  "prompt": "",
  "answer": "",
  "citations": [],
  "latency": {}
}
```

#### 3:45-5:00：基础对照与故障归因实验

- [x] 选择至少 5 个问题，分别运行无 RAG、正确上下文、错误上下文和冲突上下文，形成至少 20 条完整 Trace；
- [x] 对至少 2 个失败案例，沿 Trace 标出预期证据第一次消失或未被采用的位置；
- [x] 比较 Answer、Citation 与检索结果是否一致。

完成依据：使用固定 revision 的 Qwen3-8B 在单张 L20 上运行 20 条基础矩阵，并增加 5 条反序冲突与 5 条真实 Retriever Top-1，共得到 30 条 Trace。No RAG 5/5 拒答，Gold Context 5/5 正确，Wrong Context 5/5 传播反事实，10 条冲突输入仅 5 条拒答；真实 Top-1 的端到端可回答率为 0.8，与 Answer Chunk Recall@1 一致。已分别将 q01 Top-1 定位为 Retriever Chunk 排名失败、反事实条件定位为上游证据完整性失败、q01/q03/q05 冲突条件定位为 Generator 冲突处理失败。结果见 `results/day1_baseline.csv`、`results/day1_condition_matrix_summary.json` 与 `notes/03-transparent-dense-rag.md`。

Query Rewrite 前后对照移至第 3 天，与 Multi-query、Context Compression 和 Adaptive Routing 一起实验，避免在尚未建立基线时混入额外变量。

### 当日交付物

- 透明 Vanilla RAG 模块：`scripts/dense_retriever.py`、`scripts/context_pipeline.py`、`scripts/qwen_generator.py`；
- 条件实验入口：`scripts/run_rag_condition_matrix.py`；
- `data/eval/day1_questions.jsonl`：至少 5 个基础问题；
- `results/day1_baseline.csv`：至少 20 条带完整 Trace 的条件运行记录；
- RAG 水印传播链路图：已完成，见 `notes/01-rag-data-flow.md`。

### 验收标准

能够判断一次错误来自 Retriever、Context Packing、Prompt 还是 Generator。

### 第 1 个学习日收尾与承接

> 2026-07-23：用户决定第 1 个学习日到此结束。已经完成 RAG 数据流、原始 RAG 模型理解、上下文组织边界和参数知识冲突实验；透明 Vanilla RAG、20 条基线 Trace 与故障归因承接到第 2 个学习日。

> 2026-07-24：承接任务已经完成。透明 Dense Vanilla RAG 保存了全部关键中间对象；5 个问题的 20 条基础矩阵和 10 条诊断 Trace 已完成；Retriever、证据完整性和 Generator 冲突处理三类故障均已归因。由于用户要求服务器源码统一由本地 `scripts/` 同步，原计划的单文件 `src/rag/vanilla_rag.py` 调整为三个透明组件和一个实验入口，不降低验收标准。

下一阶段进入“第 2 天：先进检索与水印检索几何”，从 BM25 与 Dense 的同数据对照开始。

## 5. 第 2 天：先进检索与水印检索几何

### 当日目标

掌握版权水印最依赖的检索机制，量化水印查询和水印文档在不同检索器中的行为。

### 时间安排

#### 0:00-1:00：检索原理

- [x] BM25 与词项匹配；
- [x] 双塔 Dense Retrieval；
- [x] Cosine、Inner Product 与归一化；
- [x] ANN、FAISS Flat/HNSW/IVF 的基本差异；
- [x] Cross-Encoder/LLM Reranker；
- [x] RRF 融合。

> 2026-07-25：完成 FAISS Flat、HNSW 与 IVF 的两层对照。真实 32 Chunk 上，充分搜索的 HNSW 和全桶 IVF 相对 Flat 的 Top-10 Recall、Top-1 一致率及 Verification Hit@1 均为 1.0；8,192 向量压力集显示，增大 `efSearch` 或 `nprobe` 会以延迟换取 Recall。由于压力集由 32 个真实向量的高斯邻域构造，聚类结构天然有利于 IVF，不能据此给出普适算法排名。当前小语料继续使用精确 `IndexFlatIP`；完整结果见 `results/day2_faiss_ann_*`，教程见 `notes/04-advanced-retrieval-and-reranking.md` 第 5 章。

#### 1:00-3:00：实现四条检索管线

- [x] BM25；
- [x] Qwen3/BGE Dense；
- [x] BM25 + Dense + RRF；
- [x] Hybrid 全量 Top-12 + Qwen3 Reranker Top-5。

> 2026-07-24：当前受控知识库总共只有 12 个 Chunk，因此将原 Top-30 候选深度调整为全量 Top-12，而不是复制候选凑数。BM25 与 Dense 均输出完整 12 Chunk 排名，经 RRF 后使用固定 revision 的 Qwen3-Reranker-0.6B 联合打分并输出 Top-5。Answer Chunk Recall@1 从 Hybrid 的 0.8 提升到 1.0，q01 正确证据由 Rank 2 提升到 Rank 1。

#### 3:00-4:00：水印检索实验

准备至少 20 对正常/水印查询，测量：

- [x] 水印文档的 rank；
- [x] Top-1/5/10/20 命中率；
- [x] 水印文档与第 k+1 文档的分数间隔；
- [x] 正常查询误触发率；
- [x] 跨 Retriever Transfer Rate；
- [x] Reranker 前后的排名变化。

> 2026-07-24 设计纠正与重新验收：首轮事实复制型 Canary 降级为正对照。修正版使用 20 个 Normal/Trigger-only/Semantic-verification 三条件样本，Canary 不复制业务答案，并显式记录 Clean Gold Chunk。单张 L20 离线运行 60 条查询、四路全量 32 候选，共保存 240 条 Trace。四路 Normal Exact FTR@1 均为 0；BM25/Dense/RRF Trigger-only Hit@5 均为 1.0，Reranker 为 0.95；四路 Verification Hit@1 均为 1.0。`wm03` 因不含“9 个自然日”被 Reranker 从 Hybrid Rank 3 降至 Rank 7，验证了纯触发召回与证据充分过滤的区别。完整 Rank、Margin、迁移矩阵和前后变化见 `results/day2_watermark_retrieval_*`，教程见 `notes/04-advanced-retrieval-and-reranking.md` 第 4 章。

#### 4:00-5:00：消融与可视化

- [x] 水印位于句首、句中、句尾；
- [x] chunk size 为 256/512/1024；
- [x] overlap 为 0/64/128；
- [x] 使用 PCA 或 UMAP 展示查询和目标文档的位移。

> 2026-07-25：完成严格控制句子集合和字符长度的句首/句中/句尾消融。三个位置各运行 60 条三条件查询与 BM25、Dense、RRF、Qwen3 Reranker 四路全量排名，共 720 条 Trace。BM25 的 60 组 Rank、Score、Gap 在三个位置完全一致；四路 Trigger-only Hit@5 和 Verification Hit@1 在三个位置均为 1.0。Dense 的 Verification Mean Gap 从句首 0.1507 降至句尾 0.1077，Reranker 则从 4.4125 升至 5.3500，说明位置会改变排序余量，但当前强核验信号没有掉出关键 Top-k。结果见 `results/retrieval_ablation.csv` 与 `results/day2_watermark_position_*`，教程见 Day 2 笔记第 6 章。

> 2026-07-25：完成字符级 256/512/1024 Size × 0/64/128 Overlap 的九组边界压力实验与 PCA。20 份 1,800 字符长文档中的 Trigger→口令关键跨度为 142 字符，共运行 2,160 条四路 Trace。完整证据保留率从 `256/0` 的 0.50 提升到 `512/128`、`1024/128` 的 1.0；九组 Reranker Verification Hit@1 均严格等于保留率，且在保留样本内部均为 1.0，确认切分完整性构成不可由重排恢复的上游上限。PCA 使用 230 个真实 Qwen 向量，前两主成分解释约 37.5% 方差；原始 Cosine 显示更长 Chunk 会稀释 Dense 表示。结果见 `results/day2_chunking_ablation_*`、`results/retrieval_ablation.csv` 与 `results/embedding_visualization.png`，教程见 Day 2 笔记第 7、8 章。

### 当日交付物

- `retrievers/`
- `retrieval_ablation.csv`
- `embedding_visualization.png`

### 验收标准

能够用排名、分数间隔和迁移率解释某个水印为什么有效或失效。

## 6. 第 3 天：LangChain、LangGraph 与先进 RAG

### 当日目标

了解真实 RAG 系统如何改写查询、压缩上下文和动态路由，以及这些操作对水印的影响。

### 时间安排

#### 0:00-2:00：LangChain

- [ ] DocumentLoader；
- [ ] TextSplitter；
- [ ] Embedding 与 VectorStore；
- [ ] Retriever 接口和自定义 Retriever；
- [ ] LCEL/Runnable；
- [ ] MultiQuery Retriever；
- [ ] Contextual Compression；
- [ ] Parent Document Retriever；
- [ ] Metadata Filter；
- [ ] Callback 与 Trace。

要求：将第 1、2 天的自定义 Retriever 接入 LangChain，而不是重新使用黑盒组件。

#### 2:00-3:30：LangGraph Adaptive RAG

实现以下状态机：

```text
查询
→ 判断是否检索
→ 检索
→ 相关性判断
   ├── 足够：生成
   ├── 不足：重写查询后再次检索
   └── 冲突：拒答或证据核验
```

- [ ] 保存每个节点输入和输出；
- [ ] 限制最大循环次数；
- [ ] 记录水印在哪个节点丢失。

#### 3:30-4:30：先进 RAG 概览

- [ ] Self-RAG：按需检索与自反思；
- [ ] CRAG：检索结果纠错；
- [ ] RAPTOR：层次化聚类与摘要；
- [ ] GraphRAG：实体、关系、社区报告；
- [ ] Agentic RAG：LLM 自主选择检索工具。

本周只理解其机制与攻击面，不进行完整复现。

#### 4:30-5:00：攻击面表

分析以下组件是否会成为天然去水印器：

- [ ] Query Rewrite，并比较改写前后的检索 Trace；
- [ ] Multi-query；
- [ ] HyDE；
- [ ] Context Compression；
- [ ] Reranker；
- [ ] Adaptive Router；
- [ ] Graph/Hierarchy Summarization。

### 当日交付物

- `langchain_rag.py`
- `adaptive_rag.py`
- `orchestration_attack_surface.md`

### 验收标准

能在 LangChain/LangGraph 中替换 Retriever、Reranker、Prompt、Generator 和验证器。

## 7. 第 4 天：小规模复现 RAG©

### 当日目标

复现所提供论文的核心机制，并建立可靠的所有权验证评价流程。

> 2026-07-27 用户指定提前审计并运行论文补充代码，因此暂时从第 3 天切入第 4 天预实验；第 3 天任务不作完成标记。Contriever 检索门控已完成 NQ 100 问题，水印 target Hit@5 为 0.98、普通 target 泄漏率为 0.37、严格门控成功率为 0.32。因无 `OPENAI_API_KEY`，用户允许先运行固定 revision 的 Qwen3-8B 替代实验：普通/水印各 100 条输出，Qwen Judge 重复三次并多数票。实测 VSR=0.86、普通 target FPR=0.49、严格配对成功率=0.43、Answer Accuracy=0.82、H=0.18，配对 Wilcoxon `p=6.26×10⁻⁸`；10/20 问题尚不显著，50/100 问题显著。水印 target 检索后生成/判定命中 85/98，普通 target 检索后泄漏 35/37，另有 14 条未检索 target 的普通输出被判 Yes。三次 Judge 有 199/200 一致，但人工抽查发现明确假阴性，因此尚不能把所有下游失败归因于 Generator。代码和说明见 `scripts/RAG_C/reproduce_retrieval.py`、`scripts/RAG_C/reproduce_end_to_end.py`、`scripts/RAG_C/run_qwen_surrogate_server.sh`、`scripts/RAG_C/REPRODUCTION.md`。

### 时间安排

#### 0:00-1:00：数据准备

- [x] 从 NQ 选择 100 个验证问题；
- [x] 为每个问题准备正确答案；
- [ ] 生成两个正确但内容不同的解释；
- [x] 指定 target CoT 和 non-target CoT；
- [x] 生成 2-10 个词的罕见、语义无害水印短语。

#### 1:00-3:00：实现 RAG©-Lite

优先复现成本较低的 RAG©-L：

- [x] 测试水印问题检索 target CoT，Hit@5 为 0.98；
- [x] 测试正常问题检索 non-target CoT，Hit@5 为 0.61；
- [x] 每个验证问题注入两条目标文本；
- [x] 分别测试 Top-k 为 1/3/5/10；
- [ ] 保持最终答案正确。

#### 3:00-4:00：目标推理检测

实现并比较：

- [ ] 比较精确字符串、语义信息和推理路径三类验证信号；
- [ ] 说明显式 CoT、推理摘要和隐藏推理对黑盒验证的影响；
- [ ] 关键词和实体覆盖率；
- [ ] Sentence Embedding 相似度；
- [x] Qwen LLM Judge，三次多数票；
- [x] 对 LLM Judge 进行分层人工抽样复核；
- [ ] 改变 Judge Prompt，检查稳定性。

#### 4:00-5:00：统计验证

- [x] VSR/TPR；
- [x] FPR；
- [x] Harmfulness；
- [x] Answer Accuracy；
- [x] 配对 Wilcoxon 检验；
- [x] 比较 10/20/50/100 个验证问题时的统计功效；
- [x] 保存 95% Wilson 区间，不只报告 p-value。

### 当日交付物

- `ragc_lite/`
- `ragc_baseline.csv`
- `verification_statistics.ipynb` 或等效脚本。

### 验收标准

能够区分检索失败、生成失败和验证器失败，并分别报告结果。

## 8. 第 5 天：知识库盗用与去水印攻击

### 当日目标

从知识库盗用者视角，系统测试水印在实际数据处理管线中的存活性。

### 时间安排

#### 0:00-0:30：威胁模型

明确攻击者是否具备：

- [ ] 完整或局部知识库；
- [ ] 水印方案知识；
- [ ] Retriever/Generator 白盒访问；
- [ ] 验证 API 查询能力；
- [ ] 删除、改写、重编码和扩充权限。

#### 0:30-3:30：去水印处理

依次实验：

- [ ] Query Rephrasing；
- [ ] 删除罕见短语；
- [ ] 文档释义改写；
- [ ] PPL 过滤；
- [ ] Embedding 异常检测；
- [ ] 文档去重；
- [ ] 重新切分；
- [ ] 更换 Embedding；
- [ ] 添加 Reranker；
- [ ] BM25/Dense/Hybrid 切换；
- [ ] 知识库扩充与稀释；
- [ ] 仅窃取 10%/30%/50% 数据；
- [ ] 删除低频或孤立文档；
- [ ] LLM 输出重写；
- [ ] 仅返回短答案或隐藏 CoT。

#### 3:30-4:30：迁移性

- [ ] Contriever/BGE/Qwen3 间迁移；
- [ ] 无 Reranker到有 Reranker；
- [ ] Vanilla 到 LangChain Adaptive RAG；
- [ ] 不同 Generator；
- [ ] 不同 Temperature 和系统提示。

#### 4:30-5:00：汇总

统一报告：

```text
Watermark Survival Rate
Retrieval Transfer Rate
End-to-end VSR
Clean Utility
False Positive Rate
Stealth Detection AUC
Attack Cost
```

### 当日交付物

- `attacks/`
- `attack_matrix.yaml`
- `watermark_robustness.csv`

### 验收标准

能够回答水印失效是因为触发器被删除、目标文档未被检索、生成器未采用目标信息，还是验证器未检测到。

## 9. 第 6 天：RAG 知识库版权保护技术谱系

### 当日目标

把 RAG© 放入完整研究版图，并实现第二种不同范式的版权验证基线。

### 时间安排

#### 0:00-2:00：重点论文

- [ ] RAG©：推理路径水印；
- [ ] WARD：文本生成水印与可证明检测；
- [ ] RAG-WM：实体-关系驱动的合成知识水印；
- [ ] CanaryTrace：合成 Canary 与统计验证；
- [ ] KMW：知识注入式多比特水印；
- [ ] AQUA：多模态 RAG 知识版权保护。

阅读时统一记录：

| 字段 | 内容 |
|---|---|
| Threat model | 黑盒/白盒、完整/局部盗用 |
| Carrier | 原始文本、Canary、CoT、Embedding 等 |
| Trigger | 词法、语义、实体关系、随机密钥 |
| Signal | Token、知识、推理、引用、排名 |
| Detector | 统计检验、Judge、解码器 |
| Capacity | 单比特、多比特、身份编码 |
| Utility | 对知识与回答质量的影响 |
| Robustness | 删除、改写、稀释、迁移、清洗 |
| Security | 去除、伪造、歧义、串谋 |

#### 2:00-4:00：实现 Canary 基线

- [ ] 构造语义自然但唯一的合成知识；
- [ ] 生成对应验证查询；
- [ ] 将 Canary 注入知识库；
- [ ] 从输出中检测 Canary 知识；
- [ ] 使用二项检验或置换检验；
- [ ] 与 RAG©-Lite 比较。

#### 4:00-5:00：统一比较

比较维度：

- [ ] 有效性；
- [ ] 无害性；
- [ ] 隐蔽性；
- [ ] 验证查询数量；
- [ ] 对局部盗用的敏感度；
- [ ] 对输出清洗的鲁棒性；
- [ ] 对 Retriever/Generator 迁移的鲁棒性；
- [ ] 伪造和所有权歧义风险。

### 当日交付物

- `canary_baseline/`
- `paper_matrix.md`
- `method_comparison.csv`

### 验收标准

能够从威胁模型和攻击面出发选择方法，而不是只比较平均 VSR。

## 10. 第 7 天：研究问题、预实验与提案

### 当日目标

形成一个范围明确、可以立即开展的研究课题。

### 推荐主方向

> 现代 RAG 管线下知识库水印的跨组件迁移性与可验证性。

### 实验矩阵

| 变量 | 设置 |
|---|---|
| Retriever | BM25、Contriever、BGE-M3、Qwen3 |
| Fusion | Dense、Hybrid-RRF |
| Reranker | 无、Qwen3 Reranker |
| Orchestration | Vanilla、LangChain、Adaptive LangGraph |
| Query Processing | 原始、Rewrite、Multi-query、HyDE |
| Context Processing | 原始、Compression、Summary |
| Generator | 普通指令模型、推理模型、隐藏 CoT |
| Stolen Fraction | 10%、30%、50%、100% |
| Attack | 改写、过滤、重切分、稀释、响应清洗 |
| Watermark | RAG©、Canary、其他注入/改写方案 |

### 时间安排

#### 0:00-1:00：确定问题和假设

候选问题：

1. 基于旧 Dense Retriever 优化的水印能否迁移到现代 Hybrid + Reranker？
2. Query Rewrite 和 Context Compression 是否构成天然去水印器？
3. 不公开 CoT 时，如何进行输出无关或推理无关的版权验证？
4. 局部知识库盗用情况下，验证功效如何随盗用比例变化？
5. 能否设计同时抵抗去除和伪造的多比特知识库水印？

#### 1:00-2:00：设计实验

- [ ] 自变量；
- [ ] 对照组；
- [ ] 攻击者能力；
- [ ] 数据集；
- [ ] 模型和随机种子；
- [ ] 主要和次要指标；
- [ ] 统计检验与功效分析；
- [ ] 失败判据。

#### 2:00-4:00：运行最小预实验

推荐最小组合：

```text
RAG©-Lite 与 Canary
× Dense 与 Hybrid+Reranker
× 原始查询与 Query Rewrite
× 100% 与 30% 知识库盗用
```

#### 4:00-5:00：研究提案

完成 4-6 页 `research_proposal.md`：

1. 问题背景；
2. 现有方法缺口；
3. 威胁模型；
4. 核心假设；
5. 方法或分析框架；
6. 实验设置；
7. 指标和统计检验；
8. 预实验结果；
9. 风险、局限和下一步。

### 当日交付物

- `research_proposal.md`
- `pilot_results.csv`
- 可一条命令复现的实验配置；
- 下一阶段 4 周研究计划。

## 11. 全周统一评价指标

### 检索层

- Recall@k、MRR、nDCG；
- Watermark Retrieval Success Rate；
- 水印文档 Rank；
- 检索分数间隔；
- Cross-retriever Transfer Rate。

### 生成与验证层

- Answer Accuracy；
- Faithfulness；
- Citation Correctness；
- Verification Success Rate；
- False Positive/Negative Rate；
- p-value、效应量和置信区间。

### 版权保护层

- Watermark Capacity；
- Injection/Modification Rate；
- Harmfulness 或 Knowledge Corruption Rate；
- Stealth Detection AUC；
- Watermark Survival Rate；
- Query Efficiency；
- Verification Cost；
- Scrubbing、Spoofing 和 Ambiguity Resistance。

## 12. 必读材料

1. [RAG 原始论文](https://proceedings.neurips.cc/paper/2020/hash/6b493230-Abstract.html)
2. [Qwen3 Embedding/Reranker](https://github.com/QwenLM/Qwen3-Embedding)
3. [LangChain Retrieval](https://docs.langchain.com/oss/python/langchain/retrieval)
4. [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
5. [Self-RAG](https://openreview.net/pdf?id=hSyW5go0v8)
6. [RAPTOR](https://openreview.net/forum?id=GN921JHCRw)
7. [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
8. [RAG©：当前主线论文](../LLM/paper/2502.10440v1.pdf)
9. [WARD](https://www.sri.inf.ethz.ch/publications/jovanovic2025ward)
10. [RAG-WM](https://arxiv.org/abs/2501.05249)
11. [CanaryTrace](https://openreview.net/forum?id=UERyQwQ4zq)
12. [KMW](https://aclanthology.org/2026.findings-acl.1066.pdf)
13. [AQUA](https://arxiv.org/abs/2506.10030)

## 13. 每日收尾检查

每天最后 15 分钟执行：

- [ ] 提交或保存当天代码；
- [ ] 固定依赖、模型版本和随机种子；
- [ ] 保存完整查询与检索 Trace；
- [ ] 记录失败案例；
- [ ] 将结果加入统一 CSV；
- [ ] 写下一个新的研究问题；
- [ ] 明确次日第一个实验。

## 14. 最终验收清单

- [ ] 一个不依赖框架的透明 RAG；
- [ ] 一个 LangChain RAG；
- [ ] 一个 LangGraph Adaptive RAG；
- [ ] BM25、Dense、Hybrid、Reranker 四条检索管线；
- [ ] 一个 RAG©-Lite 实现；
- [ ] 一个 Canary 实现；
- [ ] 至少六种去水印处理；
- [ ] 统一的检索、生成和版权评价脚本；
- [ ] 一份论文对比矩阵；
- [ ] 一份包含预实验的研究提案。

完成以上内容后，即具备进入 RAG 知识库版权保护研究的实验基础。
