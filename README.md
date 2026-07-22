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
| 当前阶段 | 准备开始第 1 天 |
| 当前任务 | 理解 RAG 数据流 |
| 下一交付物 | Vanilla RAG、20 条测试样本、基础实验结果 |
| 详细计划 | [plan.md](./plan.md) |
| 研究主线 | RAG 知识库版权保护与所有权验证 |

最后更新：2026-07-22

## 7 天学习进度

| 天数 | 主题 | 状态 | 主要产出 |
|---|---|---|---|
| Day 1 | RAG 与 LLM 的完整协同机制 | 未开始 | 透明 Vanilla RAG |
| Day 2 | 先进检索与水印检索几何 | 未开始 | BM25/Dense/Hybrid/Reranker 对比 |
| Day 3 | LangChain、LangGraph 与先进 RAG | 未开始 | LangChain 与 Adaptive RAG |
| Day 4 | 小规模复现 RAG© | 未开始 | RAG©-Lite 与统计验证 |
| Day 5 | 知识库盗用与去水印攻击 | 未开始 | 攻击矩阵与鲁棒性结果 |
| Day 6 | RAG 知识库版权保护技术谱系 | 未开始 | Canary 基线与论文矩阵 |
| Day 7 | 研究问题、预实验与提案 | 未开始 | Research Proposal |

进度以 [plan.md](./plan.md) 中经过验证后勾选的任务为准。

## 计划中的仓库结构

目录将在首次需要时创建，避免没有实际内容的空目录。

```text
RAG/
├── README.md                 # 项目介绍、公开进度和使用说明
├── AGENTS.md                 # 教学角色、工作约定和持续记忆
├── plan.md                   # 7 天详细学习计划
├── configs/                  # 实验配置
├── data/                     # 小型样本、评测数据和数据说明
├── notes/                    # 每日笔记和论文笔记
├── src/
│   ├── ingestion/            # 加载、解析和切分
│   ├── retrievers/           # BM25、Dense、Hybrid
│   ├── rerankers/            # Reranker
│   ├── rag/                  # Vanilla、LangChain、Adaptive RAG
│   ├── watermark/            # RAG©、Canary 等方法
│   ├── attacks/              # 授权环境内的安全实验
│   └── evaluation/           # 检索、生成和版权指标
├── experiments/              # 可复现实验入口
├── results/                  # 小型结果、图表和摘要
├── tests/                    # 自动化测试
└── references/               # 论文索引和外部资料说明
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

当前仓库处于学习准备阶段。代码、依赖和运行命令将在 Day 1 开始实现后补充。所有实验将尽量记录：

- Python 与依赖版本；
- 数据来源和许可证；
- 模型、Prompt 和随机种子；
- Retriever 返回结果和分数；
- LLM 输入、输出和验证信号；
- 评价指标、置信区间和失败案例。

## 仓库维护

- [AGENTS.md](./AGENTS.md) 描述 Codex 在本仓库中的教师和维护者职责；
- 每完成一个经过验证的任务，同步更新 `plan.md`；
- 每完成一个学习日或重要实验，同步更新本 README；
- 大型数据、模型权重、缓存、临时文件和密钥不会提交到仓库；
- 实验仅在自有或明确授权的环境中开展。

## 最近更新

- 2026-07-22：建立 7 天学习计划；
- 2026-07-22：明确 Codex 的教师、进度维护和仓库整理职责；
- 2026-07-22：创建公开进度 README。
- 2026-07-22：初始化 Git `main` 分支并绑定远程仓库。
