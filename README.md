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
| 当前阶段 | 第 1 天进行中 |
| 当前任务 | Day 1 精选资源导读与学习总结 |
| 下一交付物 | Vanilla RAG、20 条测试样本、基础实验结果 |
| 详细计划 | [plan.md](./plan.md) |
| 研究主线 | RAG 知识库版权保护与所有权验证 |
| Obsidian | Vault 与知识地图已就绪 |

最后更新：2026-07-22

## 7 天学习进度

| 天数 | 主题 | 状态 | 主要产出 |
|---|---|---|---|
| Day 1 | RAG 与 LLM 的完整协同机制 | 进行中 | 透明 Vanilla RAG |
| Day 2 | 先进检索与水印检索几何 | 未开始 | BM25/Dense/Hybrid/Reranker 对比 |
| Day 3 | LangChain、LangGraph 与先进 RAG | 未开始 | LangChain 与 Adaptive RAG |
| Day 4 | 小规模复现 RAG© | 未开始 | RAG©-Lite 与统计验证 |
| Day 5 | 知识库盗用与去水印攻击 | 未开始 | 攻击矩阵与鲁棒性结果 |
| Day 6 | RAG 知识库版权保护技术谱系 | 未开始 | Canary 基线与论文矩阵 |
| Day 7 | 研究问题、预实验与提案 | 未开始 | Research Proposal |

进度以 [plan.md](./plan.md) 中经过验证后勾选的任务为准。

## 仓库结构

当前仓库结构如下；代码和实验目录将在首次实际使用时创建。

```text
RAG/
├── README.md                 # 项目介绍、公开进度和使用说明
├── AGENTS.md                 # 教学角色、工作约定和持续记忆
├── plan.md                   # 7 天详细学习计划
├── .obsidian/                # 可共享的 Vault 设置
├── notes/
│   ├── 00-RAG知识地图.md      # Obsidian 主入口
│   ├── 01-rag-data-flow.md   # RAG 数据流与 LLM 协同
│   ├── assets/               # 笔记引用的小型图片和附件
│   ├── concepts/             # 原子概念笔记
│   ├── papers/               # 论文阅读笔记
│   ├── experiments/          # 实验记录
│   ├── research-ideas/       # 研究问题
│   └── templates/            # Obsidian 知识章节与研究模板
└── .gitignore                # 缓存、临时文件和大型产物规则
```

## Obsidian 使用

在 Obsidian 中选择 **Open folder as vault**，打开本仓库根目录。主入口是 [RAG 知识地图](./notes/00-RAG知识地图.md)。

当前共享设置包括：

- 新笔记默认放入 `notes/`；
- 附件默认放入 `notes/assets/`；
- 使用标准 Markdown 链接；
- 自动更新内部链接；
- 按学习阶段组织的知识章节直接放入 `notes/`，正文不记录进度轨迹；
- 模板目录为 `notes/templates/`；
- Git、缓存、模型和本地大型数据不进入 Obsidian 搜索。

个人窗口布局、主题、插件状态和 Graph 显示偏好不会提交到 Git。

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
- [RAG 知识地图](./notes/00-RAG知识地图.md) 是 Obsidian 学习入口；
- 每完成一个经过验证的任务，同步更新 `plan.md`；
- 每完成一个学习日或重要实验，同步更新本 README；
- 大型数据、模型权重、缓存、临时文件和密钥不会提交到仓库；
- 实验仅在自有或明确授权的环境中开展。

## 最近更新

- 2026-07-22：整理仓库结构，将知识章节直接放入 `notes/`、附件迁入 `notes/assets/`，并清理 Daily Notes 配置与临时 PDF 产物；
- 2026-07-22：建立 7 天学习计划；
- 2026-07-22：明确 Codex 的教师、进度维护和仓库整理职责；
- 2026-07-22：创建公开进度 README；
- 2026-07-22：初始化 Git `main` 分支并绑定远程仓库；
- 2026-07-22：将初始学习计划与维护文档推送至 `origin/main`；
- 2026-07-22：建立 Obsidian Vault、知识地图、Day 1 笔记、论文笔记和模板体系。
