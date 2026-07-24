# RAG Learning & Knowledge Base Copyright Protection

这是一个面向现代 Retrieval-Augmented Generation（RAG）技术学习与知识库版权保护研究的持续实践仓库。

仓库首先建立透明、可复现的 RAG 实验基础，随后研究知识库水印在现代 Retriever、Reranker、LLM 和编排框架中的有效性、迁移性与鲁棒性。

## 研究目标

- 掌握现代 RAG 的完整技术链路；
- 理解 Retriever、Reranker、Prompt、LLM 与编排框架的协同；
- 使用 Python 实现可替换、可追踪、可评测的 RAG；
- 学习 LangChain、LangGraph、Adaptive RAG 等常用技术；
- 复现并分析 RAG 知识库版权保护方法；
- 研究水印去除、伪造、局部盗用和跨组件迁移问题。

## 背景

本项目建立在已有的开源数据集版权保护、后门水印和后门攻击研究经验之上。学习重点不是通用后门基础，而是 RAG 特有的信息流和版权保护问题：

```text
知识库
→ 文档解析与切分
→ Embedding 与索引
→ Retriever
→ Reranker
→ Context Packing
→ Prompt/LLM
→ 输出与所有权验证
```

## 当前状态

| 项目 | 状态 |
|---|---|
| 当前阶段 | 第 2 个学习日：先进检索与水印检索几何 |
| 当前任务 | 水印位于句首、句中、句尾的检索消融 |
| 下一交付物 | `retrieval_ablation.csv` 与 `embedding_visualization.png` |
| 详细计划 | [plan.md](./plan.md) |
| 研究主线 | RAG 知识库版权保护与所有权验证 |

最后更新：2026-07-25

## 7 天学习进度

| 天数 | 主题 | 状态 | 主要产出 |
|---|---|---|---|
| Day 1 | RAG 与 LLM 的完整协同机制 | 已完成 | 透明 Vanilla RAG、30 条 Trace 与故障归因 |
| Day 2 | 先进检索与水印检索几何 | 进行中 | 四路检索、水印迁移与 FAISS ANN 对比 |
| Day 3 | LangChain、LangGraph 与先进 RAG | 未开始 | LangChain 与 Adaptive RAG |
| Day 4 | 小规模复现 RAG© | 未开始 | RAG©-Lite 与统计验证 |
| Day 5 | 知识库盗用与去水印攻击 | 未开始 | 攻击矩阵与鲁棒性结果 |
| Day 6 | RAG 知识库版权保护技术谱系 | 未开始 | Canary 基线与论文矩阵 |
| Day 7 | 研究问题、预实验与提案 | 未开始 | Research Proposal |

进度以 [plan.md](./plan.md) 中经过验证后勾选的任务为准。

## 仓库结构

当前仓库结构如下：

```text
RAG/
├── README.md                 # 项目介绍、公开进度和使用说明
├── AGENTS.md                 # 教学角色、工作约定和持续记忆
├── plan.md                   # 7 天详细学习计划
├── requirements-server.txt  # 远程 GPU 环境的已验证依赖
├── data/
│   ├── README.md             # 合成数据来源、许可与再生成说明
│   ├── clean/               # 受控虚构知识库
│   ├── watermarked/         # 可追踪的水印检索实验语料
│   └── eval/                # 基础问题与三条件水印查询组
├── notes/
│   ├── 00-RAG知识地图.md      # 笔记导航入口
│   ├── 01-rag-data-flow.md   # RAG 数据流与 LLM 协同
│   ├── 02-ragc-paper.md      # RAG© 论文笔记
│   ├── 03-transparent-dense-rag.md # 透明 Dense RAG 教程与实验结果
│   ├── 04-advanced-retrieval-and-reranking.md # BM25、Hybrid 与重排教程
│   └── assets/               # 笔记引用的小型图片和附件
├── scripts/
│   ├── bm25_retriever.py     # 可解释中文 BM25 Retriever
│   ├── build_chunks.py      # 带字符位置和验证的透明切分器
│   ├── context_pipeline.py  # Context Packing 与 Prompt Builder
│   ├── dense_retriever.py   # Qwen3 + FAISS 可复用 Dense Retriever
│   ├── download_server_models.py # 固定 revision 的服务器模型下载
│   ├── qwen_generator.py    # 单卡 Qwen3 Generator 与输出解析
│   ├── qwen_reranker.py     # Query–Chunk 联合打分的 Qwen3 Reranker
│   ├── rrf_fusion.py        # 可解释 Reciprocal Rank Fusion
│   ├── watermark_retrieval_metrics.py # Rank、Margin、误触发与迁移指标
│   ├── build_watermark_retrieval_dataset.py # 20 组三条件水印数据生成
│   ├── run_context_packing.py # Top-1/Top-2 Prompt 对照实验
│   ├── run_bm25_retrieval.py # BM25 评测与 Dense 对照
│   ├── run_dense_retrieval.py # 5 问题 Top-k 检索评测
│   ├── run_faiss_ann_comparison.py # Flat/HNSW/IVF 两层对照
│   ├── run_qwen_generator_probe.py # q01 Top-1/Top-2 生成对照
│   ├── run_qwen_reranker.py # 全量 Hybrid 候选重排实验
│   ├── run_watermark_retrieval_experiment.py # 四管线水印迁移实验
│   ├── run_rag_condition_matrix.py # 5 问题的 30 条生成条件矩阵
│   ├── run_rrf_hybrid_retrieval.py # BM25 + Dense RRF 对照实验
│   ├── run_server_python.sh # 约束服务器缓存和临时目录
│   └── smoke_dense_retrieval.py # Qwen3 + FAISS 冒烟实验
├── results/                 # 切分、检索、水印与 FAISS 对照结果
├── tests/                   # 检索组件与实验指标的自动化测试
└── .gitignore                # 缓存、临时文件和大型产物规则
```

## 重点研究问题

1. 基于旧 Dense Retriever 优化的水印能否迁移到现代 Embedding、Hybrid Retrieval 和 Reranker？
2. Query Rewrite、Multi-query、Context Compression 和 Adaptive Routing 是否构成天然去水印器？
3. 推理模型不公开真实 Chain-of-Thought 时，如何实现可靠的黑盒版权验证？
4. 局部知识库盗用、重切分、扩充和响应清洗如何影响验证功效？
5. 如何同时抵抗水印去除、伪造和所有权歧义？

## 主要参考资料

- [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper/2020/hash/6b493230-Abstract.html)
- [Qwen3 Embedding/Reranker](https://github.com/QwenLM/Qwen3-Embedding)
- [LangChain Retrieval](https://docs.langchain.com/oss/python/langchain/retrieval)
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)
- [Self-RAG](https://openreview.net/pdf?id=hSyW5go0v8)
- [RAPTOR](https://openreview.net/forum?id=GN921JHCRw)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [RAG©: Towards Copyright Protection for Knowledge Bases](https://arxiv.org/abs/2502.10440)
- [WARD](https://www.sri.inf.ethz.ch/publications/jovanovic2025ward)
- [RAG-WM](https://arxiv.org/abs/2501.05249)
- [CanaryTrace](https://openreview.net/forum?id=UERyQwQ4zq)
- [Knowledge-Infused Multi-Bit Watermarking](https://aclanthology.org/2026.findings-acl.1066.pdf)

## 复现说明

当前已完成透明 Dense RAG、BM25、RRF Hybrid 和 Qwen3 Reranker。纠正后的水印实验使用 20 组普通业务查询、仅增加触发词的控制查询和专用语义验证查询，Canary 不复制业务答案。四路 Normal Exact FTR@1 均为 0；Trigger-only Hit@5 在 BM25/Dense/RRF/Reranker 上分别为 1.0/1.0/1.0/0.95；Verification Hit@1 四路均为 1.0。教程与复现命令见 [透明 Dense RAG：从文档切分到证据约束生成](./notes/03-transparent-dense-rag.md) 和 [先进检索与重排：从 BM25 到 Hybrid RAG](./notes/04-advanced-retrieval-and-reranking.md)。实验持续记录：

- Python 与依赖版本；
- 数据来源和许可证；
- 模型、Prompt 和随机种子；
- Retriever 返回结果和分数；
- LLM 输入、输出和验证信号；
- 评价指标、置信区间和失败案例。

## 仓库维护

- [AGENTS.md](./AGENTS.md) 描述 Codex 在本仓库中的教师和维护者职责；
- [RAG 知识地图](./notes/00-RAG知识地图.md) 是 Markdown 笔记导航入口；
- 每完成一个经过验证的任务，同步更新 `plan.md`；
- 每完成一个学习日或重要实验，同步更新本 README；
- 大型数据、模型权重、缓存、临时文件和密钥不会提交到仓库；
- 实验仅在自有或明确授权的环境中开展。

## 最近更新

- 2026-07-25：完成 FAISS Flat/HNSW/IVF 两层对照；真实 32 Chunk 上三种索引保持相同 Top-10 与 Verification Hit@1，8,192 向量压力集验证 `efSearch`/`nprobe` 的速度—召回权衡；当前小语料继续使用精确 `IndexFlatIP`；
- 2026-07-24：完成纠正后的 20 组三条件水印检索实验；60 条查询形成 240 条四路 Trace，Normal Exact FTR@1 四路均为 0，Trigger-only Hit@5 经 Reranker 从 1.0 降至 0.95，Verification Hit@1 四路均为 1.0；
- 2026-07-24：复核后确认首轮事实复制型 Canary 不能作为有效水印实验，将其降级为正对照；本地数据已纠正为 20 组三条件查询，Canary 不再复制业务答案，并显式记录 Clean Gold Chunk；
- 2026-07-24：完成固定 revision 的 Qwen3-Reranker-0.6B 全量 Top-12 → Top-5 实验；q01 正确证据由 Rank 2 升至 Rank 1，Answer Recall@1 与 MRR 均达到 1.0；记录未校准概率饱和、Logit Margin、离线模型哈希与运行成本，教程新增第三章；
- 2026-07-24：完成 BM25 + Dense 的透明 RRF Hybrid；Hybrid Gold Answer Chunk Recall@1 为 0.8，q01、q02、q05 出现对称 Rank 精确并列，确认融合不保证优于最佳单路；Day 2 教程新增 RRF 完整章节；
- 2026-07-24：完成透明中文 BM25 与 Dense 同数据对照；5 个问题的 Gold Answer Chunk Recall@1 为 1.0，修复 Dense 的 q01 Rank-2 证据问题，两者 Top-1 Chunk 一致率为 0.4；新增 Day 2 教程首章；
- 2026-07-24：完成 20 条基础上下文矩阵和 10 条附加诊断 Trace；No RAG 5/5 拒答、Gold 5/5 正确、Wrong 5/5 传播反事实，冲突证据仅 5/10 拒答，并发现内容相关的顺序效应；
- 2026-07-24：完成 Retriever 排名、上游证据完整性和 Generator 冲突处理三类故障归因，Day 1 透明 Vanilla RAG 基线闭环；
- 2026-07-24：整理两天实验代码与结果，新增透明 Dense RAG 教程，串联参数知识冲突、可追踪切分、Qwen3 + FAISS、证据级评测、Context Packing、Qwen3-8B 生成和 Trace 故障归因；
- 2026-07-23：固定 Qwen3-Embedding 与 Qwen3-8B revision，将模型缓存迁入 `/data/haojiachen/rag/models` 并完成离线加载验证；`q01` Top-1 拒答、Top-2 正确回答和引用；
- 2026-07-23：完成 Top-1/Top-2 Context Packing 与 Prompt Builder；确认 `q01` 证据在 Top-1 缺失、Top-2 进入 Prompt；
- 2026-07-23：完成 12 个 Chunk 的 Qwen3 + FAISS 基线；Gold Document Recall@1 为 1.0，Gold Answer Chunk Recall@1 为 0.8，发现首个 Chunk-level 检索失败案例；
- 2026-07-23：构造 5 份受控文档与 5 个问题，生成并验证 12 个可追踪 Chunk；Qwen3-Embedding + FAISS 服务器冒烟实验通过；
- 2026-07-23：第 1 个学习日收尾；下一次先承接透明 Dense RAG、20 条 Trace 和故障归因，再进入先进检索；
- 2026-07-22：完成 Gemini 参数知识与 Retrieved Context 冲突实验，记录 System 指令、上下文顺序和拒答行为；
- 2026-07-22：完成 Day 1 RAG 数据流模块，后续集中进行知识冲突实验、透明 Dense RAG 和故障归因；
- 2026-07-22：整理仓库结构，将知识章节直接放入 `notes/`、附件迁入 `notes/assets/`，并清理旧配置与临时 PDF 产物；
- 2026-07-22：建立 7 天学习计划；
- 2026-07-22：明确 Codex 的教师、进度维护和仓库整理职责；
- 2026-07-22：创建公开进度 README；
- 2026-07-22：初始化 Git `main` 分支并绑定远程仓库；
- 2026-07-22：将初始学习计划与维护文档推送至 `origin/main`；
