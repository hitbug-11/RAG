# 先进检索与重排：从 BM25 到 Hybrid RAG

本笔记围绕同一批 Chunk 和问题，逐步实现并比较 Sparse、Dense、Hybrid、Reranked Retrieval 与不同 FAISS 索引。当前已经建立透明 BM25、RRF Hybrid 和 Qwen3 Reranker 管线，完成水印实验设计纠错，并通过真实向量与压力集对照解释 Flat、HNSW、IVF 的准确率—延迟权衡。

## 知识点速查

- [1. 透明 BM25 与 Dense 对照实验](#1-透明-bm25-与-dense-对照实验)
- [2. BM25 + Dense 的 RRF Hybrid 实验](#2-bm25--dense-的-rrf-hybrid-实验)
- [3. Qwen3 Reranker 全量候选重排实验](#3-qwen3-reranker-全量候选重排实验)
- [4. 事实复制型 Canary 正对照与设计纠错](#4-事实复制型-canary-正对照与设计纠错)
- [5. FAISS Flat、HNSW、IVF 对照实验](#5-faiss-flathnswivf-对照实验)
- [小结](#小结)
- [参考资料](#参考资料)

## 1. 透明 BM25 与 Dense 对照实验

### 1.1 知识定位：Sparse 与 Dense 在比较什么

BM25 和 Dense Retrieval 接收相同的 Query 与 Chunk，却用两套完全不同的相关性证据：

| 检索器 | 比较对象 | 擅长 | 典型弱点 |
|---|---|---|---|
| BM25 | Query 与 Chunk 的离散词项 | 精确术语、编号、专名、罕见短语 | 同义改写、没有字面重合的语义匹配 |
| Dense | Query 与 Chunk 的连续向量 | 语义近似、自然语言改写 | 可能把同主题但不含答案的 Chunk 排在前面 |

本实验不替换数据，只替换 Retriever。输入仍是 Day 1 生成的 12 个 Chunk 和 5 个问题，Dense 对照仍是固定 Revision 的 `Qwen3-Embedding-0.6B + FAISS IndexFlatIP` 结果。因此排名差异可以归因到检索机制，而不是切分或问题变化。

### 1.2 实验数据流

```mermaid
flowchart LR
    subgraph Offline["离线 BM25 建库"]
        C["12 个相同 Chunk"] --> TOK["NFKC 归一化<br/>中文 unigram + bigram"]
        TOK --> TF["每个 Chunk 的 TF 与长度"]
        TF --> DF["全库 DF / IDF"]
    end

    subgraph Online["在线打分"]
        Q["Query"] --> QT["同一 Tokenizer"]
        QT --> SCORE["BM25 逐词项贡献"]
        TF --> SCORE
        DF --> SCORE
        SCORE --> TOPK["Top-5 Chunk + 分数解释"]
    end

    subgraph Evaluation["同标注评测"]
        TOPK --> RANK["Gold Document Rank<br/>Gold Answer Chunk Rank"]
        DENSE["已保存的 Dense Top-5"] --> CMP["逐问题排名对照"]
        RANK --> CMP
    end
```

离线阶段只统计词项，不加载神经网络；在线阶段也不计算向量。实验为每个结果保留总分和贡献最大的词项，使“为什么排在这里”可以直接检查。

### 1.3 先用直觉理解 BM25

可以把 BM25 看成三条规则：

1. Query 中的词在某个 Chunk 出现，才会贡献分数；
2. 全库越少见的词，区分能力越强，权重越大；
3. 同一个词在 Chunk 中重复出现会继续加分，但收益逐渐饱和；过长 Chunk 还会受到长度归一化。

例如查询包含“无理由退款”时，含有“无理”“理由”“由退”等 bigram 的退款期限 Chunk 会获得直接贡献；只谈“退款申请”但不含这些短语的同主题 Chunk，也可能相关，但少了这些精确匹配分数。

### 1.4 符号与公式

对查询 \(Q\) 和文档 Chunk \(D\)，本实验使用：

\[
\operatorname{BM25}(D,Q)
=
\sum_{t \in Q}
\operatorname{IDF}(t)
\cdot
\frac{f(t,D)(k_1+1)}
{f(t,D)+k_1\left(1-b+b\frac{|D|}{\operatorname{avgdl}}\right)}
\]

其中：

- \(t\)：Query 中的一个词项；
- \(f(t,D)\)：词项 \(t\) 在 Chunk \(D\) 中的出现次数；
- \(|D|\)：该 Chunk 的词项数；
- \(\operatorname{avgdl}\)：全库 Chunk 的平均词项数；
- \(k_1=1.5\)：控制词频饱和速度；
- \(b=0.75\)：控制长度归一化强度。

IDF 使用非负的 Robertson 变体：

\[
\operatorname{IDF}(t)
=
\log\left(
1+\frac{N-n_t+0.5}{n_t+0.5}
\right)
\]

\(N\) 是 Chunk 总数，\(n_t\) 是包含词项 \(t\) 的 Chunk 数。若一个词几乎到处出现，\(n_t\) 较大，IDF 就小；罕见短语的 \(n_t\) 较小，IDF 就大。

需要特别注意：BM25 分数只在同一个索引和同一组参数内部有排序意义，不能把 `35.07` 与 Dense Cosine 分数 `0.77` 直接比较大小。

### 1.5 中文 Tokenizer：为什么同时使用 unigram 和 bigram

英文通常可先按空格得到词；中文没有天然空格。为了让实验透明且不依赖外部分词模型，本实现先做 NFKC 与小写归一化，再使用：

- 中文字符 unigram：提高召回，例如“退”“款”；
- 相邻字符 bigram：保留局部短语，例如“退款”“无理”“理由”；
- 连续英文和数字：作为一个词项，例如 `qwen3-8b`。

核心代码位于 [`scripts/bm25_retriever.py`](../scripts/bm25_retriever.py)：

```python
def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    terms = []
    for match in TOKEN_PATTERN.finditer(normalized):
        span = match.group()
        if "\u4e00" <= span[0] <= "\u9fff":
            terms.extend(span)
            terms.extend(
                span[index:index + 2]
                for index in range(len(span) - 1)
            )
        else:
            terms.append(span)
    return terms
```

这种方法不是唯一正确的中文切词方案。它的优势是确定、无需词典、便于分析短触发词；局限是 unigram 会放大常见单字，bigram 也不等价于真实词边界。后续水印实验应把 Tokenizer 作为显式变量，而不是把它藏在 Retriever 内部。

### 1.6 建库与逐词项打分

初始化时先为每个 Chunk 保存词频、长度和全库文档频率：

```python
self.term_frequencies = [
    Counter(tokenize(chunk["text"]))
    for chunk in chunks
]
self.document_lengths = [
    sum(frequencies.values())
    for frequencies in self.term_frequencies
]
self.average_document_length = (
    sum(self.document_lengths) / len(self.document_lengths)
)

for frequencies in self.term_frequencies:
    self.document_frequencies.update(frequencies.keys())
```

打分时不只累加总分，还保存每个命中词项的 `query_tf`、`document_tf`、`document_frequency`、`idf` 和 `contribution`：

```python
contribution = (
    query_frequency
    * inverse_document_frequency
    * term_frequency
    * (self.k1 + 1)
    / (term_frequency + length_normalization)
)
```

最终 Trace 因而既回答“排了第几”，也能回答“哪些词贡献了这个排名”。这比只调用一个 BM25 库并读取总分更适合机制研究。

### 1.7 运行与自动验证

BM25 本身只依赖 Python 标准库。本地复现命令为：

```bash
PYTHONPATH=scripts python -m unittest discover -s tests -v
PYTHONPATH=scripts python scripts/run_bm25_retrieval.py
```

4 个单元测试验证了：

1. 中文 unigram/bigram 与英文归一化结果；
2. 罕见词项的 IDF 高于常见词项；
3. 精确短语匹配能够排到第一，并可读取贡献解释；
4. 非法 `top_k` 会被拒绝。

实验产物包括：

- [`day2_bm25_retrieval.jsonl`](../results/day2_bm25_retrieval.jsonl)：5 个问题的完整 Top-5、分数与词项贡献；
- [`day2_bm25_retrieval_summary.json`](../results/day2_bm25_retrieval_summary.json)：参数、语料统计和汇总指标；
- [`day2_bm25_dense_comparison.csv`](../results/day2_bm25_dense_comparison.csv)：逐问题 Dense/BM25 排名对照。

### 1.8 运行结果

索引包含 12 个 Chunk，Tokenizer 产生 976 个不同词项，平均 Chunk 长度为 182.583 个词项。5 个问题的汇总结果如下：

| 指标 | BM25 | Dense |
|---|---:|---:|
| Gold Document Recall@1 | 1.0 | 1.0 |
| Gold Answer Chunk Recall@1 | 1.0 | 0.8 |
| Gold Answer Chunk Recall@3 | 1.0 | 1.0 |
| Gold Answer Chunk MRR | 1.0 | 0.9 |

两种检索器的 Top-1 Chunk 一致率为 0.4，即 5 个问题中只有 2 个选择了同一个首位 Chunk。

逐问题排名为：

| 问题 | BM25 Top-1 | Dense Top-1 | BM25 答案 Rank | Dense 答案 Rank |
|---|---|---|---:|---:|
| q01 退款 | `refund#chunk-001` | `refund#chunk-000` | 1 | 2 |
| q02 发票 | `invoice#chunk-000` | `invoice#chunk-001` | 1 | 1 |
| q03 会员 | `membership#chunk-001` | `membership#chunk-001` | 1 | 1 |
| q04 保修 | `warranty#chunk-000` | `warranty#chunk-000` | 1 | 1 |
| q05 物流 | `logistics#chunk-000` | `logistics#chunk-001` | 1 | 1 |

表中的 Chunk 名为便于阅读的缩写，完整稳定 ID 保存在 CSV 与 JSONL 中。两种检索器只有 q03、q04 的 Top-1 完全相同；q02、q05 虽选择了不同 Chunk，但由于 overlap 复制了答案，两者的答案 Rank 都是 1。

### 1.9 重点案例：BM25 为什么修复 q01

q01 查询是“青岚商城普通商品签收后多久可以申请无理由退款？”。BM25 的前三名为：

| Rank | Chunk | 是否含答案 | BM25 分数 |
|---:|---|---|---:|
| 1 | `qinglan-refund-v1#chunk-001` | 是，含“9 个自然日” | 35.066793 |
| 2 | `qinglan-refund-v1#chunk-000` | 否 | 29.597531 |
| 3 | `qinglan-refund-v1#chunk-002` | 否 | 25.620535 |

Rank 1 中贡献最大的词项包括“无”“无理”“理由”“由退”和“品”，前四项各贡献 2.228075。该 Chunk 同时出现“普通商品”“物流签收”和“申请无理由退款”，与查询形成密集的字面重合。

Dense Retrieval 把 `chunk-000` 排在第一。它正确识别了“青岚商城、普通商品、退款申请、签收”等整体主题，但这个 Chunk 只说明申请材料，不含退款期限。换言之：

```text
Dense：主题非常相关 → Rank 1，但证据不完整
BM25：问题关键短语精确重合 → 含答案 Chunk Rank 1
```

这不是“BM25 普遍优于 Dense”的证据。当前查询和政策正文使用了高度相似的措辞，天然有利于词项检索；若把查询改写为“收到货后几天内能反悔退货”，BM25 可能因字面重合下降而退化，而 Dense 仍可能识别其语义。

### 1.10 与知识库版权保护的关系

BM25 对知识库水印的影响可以直接从公式解释：

- 含罕见触发短语的水印 Chunk 会获得较高 IDF，容易被精确触发；
- 触发短语被同义改写、拆分或规范化后，词项重合可能迅速消失；
- 水印词重复过多不会线性增加分数，因为词频贡献会饱和；
- Chunk 过长会受到长度归一化惩罚；
- overlap 可能复制水印，使多个 Chunk 同时命中，也可能造成所有权信号重复计数。

因此，“水印能被 Dense Retriever 找到”不能推出它也能迁移到 BM25，反之亦然。后续跨 Retriever Transfer Rate 应在同一批正常/水印查询上分别测量目标 Chunk Rank，再通过 Hybrid 与 Reranker 观察信号是被保留、提升还是过滤。

### 1.11 实验边界与易错点

- 只有 12 个 Chunk 和 5 个措辞接近正文的问题，结果用于验证机制，不代表真实语料质量；
- 中文 unigram/bigram 会让词项长度大于原始字符数，不能和其他 Tokenizer 的 `avgdl` 直接比较；
- BM25 分数与 Dense Cosine 分数不在同一量纲，Hybrid 时不能直接相加；
- 当前 Gold Answer 判断依赖答案字符串别名，尚未覆盖语义等价表达；
- 本实验还没有正常/水印查询对，因此不能据此报告水印迁移率或误触发率；
- RRF 融合和 Reranker 尚未加入，本章结论只覆盖第一阶段候选召回。

## 2. BM25 + Dense 的 RRF Hybrid 实验

### 2.1 知识定位：为什么不能直接相加原始分数

BM25 与 Dense 的分数来自不同空间：

- BM25 分数由 TF、IDF 和长度归一化累加，本实验约为 `3～61`；
- Dense 分数是归一化向量的 Inner Product，即 Cosine Similarity，本实验约为 `0.2～0.9`。

直接计算 `BM25 score + Dense score` 会让数值范围更大的 BM25 主导结果；即使先做 Min-Max 或 Z-score，也会引入候选集合和分布依赖。RRF 避开了这个问题：它完全忽略原始分数，只使用每个 Retriever 给出的名次。

因此 RRF 解决的是“异构分数如何融合”，而不是“如何判断某个 Retriever 更可信”。若需要体现来源可靠性，还要引入 Weighted RRF、学习排序或后续 Reranker。

### 2.2 融合数据流

```mermaid
flowchart LR
    Q["同一个 Query"] --> BM["BM25 Top-5<br/>Chunk ID + Rank"]
    Q --> DE["Dense Top-5<br/>Chunk ID + Rank"]
    BM --> UNION["按稳定 Chunk ID<br/>构造候选并集"]
    DE --> UNION
    UNION --> CONTRIB["每个来源贡献<br/>1 / (k + rank)"]
    CONTRIB --> SUM["按 Chunk 求和"]
    SUM --> SORT["RRF Score 排序<br/>显式并列规则"]
    SORT --> HY["Hybrid Top-5"]
    HY --> EVAL["Answer Rank / Recall / MRR<br/>来源名次与并列诊断"]
```

这个数据流依赖稳定 `chunk_id`。若 BM25 与 Dense 为同一段文本生成了不同 ID，RRF 会把它们误认为两个候选，无法累加共识；若不同文本错误复用同一 ID，又会把不相干内容合并。本实现因此同时校验共享 Chunk 的正文、文档 ID、字符位置和 Metadata。

### 2.3 先用直觉理解 RRF

RRF 遵循两条简单规则：

1. 在单个 Retriever 中排得越靠前，贡献越大；
2. 同时被多个 Retriever 找到的 Chunk，会累加多个来源的贡献。

对候选 Chunk \(d\)，两个来源的 RRF 分数为：

\[
\operatorname{RRF}(d)
=
\frac{1}{k+r_{\text{BM25}}(d)}
+
\frac{1}{k+r_{\text{Dense}}(d)}
\]

若某个来源的 Top-k 中没有 \(d\)，该来源贡献为 0。这里：

- \(r_s(d)\)：Chunk \(d\) 在来源 \(s\) 中的 Rank；
- \(k=60\)：平滑常数，减小 Rank 1 与后续名次之间的绝对差距；
- 本实验每个来源输入 Top-5，输出 Hybrid Top-5。

例如一个 Chunk 在两个来源都排 Rank 1：

\[
\frac{1}{60+1}+\frac{1}{60+1}
=0.032786885
\]

若它只在一个来源排 Rank 1，则分数只有：

\[
\frac{1}{60+1}
=0.016393443
\]

因此“双来源共识”通常能够超过“单来源高排名”。

### 2.4 对称 Rank 交换为什么会精确并列

q01 中两个退款 Chunk 的来源排名正好互换：

| Chunk | BM25 Rank | Dense Rank | RRF |
|---|---:|---:|---:|
| `refund#chunk-000` | 2 | 1 | \(1/62 + 1/61=0.032522475\) |
| `refund#chunk-001` | 1 | 2 | \(1/61 + 1/62=0.032522475\) |

加法满足交换律，所以两个分数必然相同。调节 \(k\) 不会解除这种对称并列，只会同时改变两者的共同分数。

这揭示了 RRF 的一个关键边界：它知道两个 Retriever 的名次，却不知道哪一个 Retriever 对当前问题更可靠，也不知道哪个 Chunk 真正包含答案。因此，RRF 必须配合确定性的并列规则；但这个规则只能保证复现，不能凭空增加相关性信息。

### 2.5 核心实现：只融合 Rank，保留来源贡献

核心代码位于 [`scripts/rrf_fusion.py`](../scripts/rrf_fusion.py)。每个来源先验证 Rank 从 1 连续排列，再按稳定 Chunk ID 累加贡献：

```python
for source_name, results in rankings.items():
    for expected_rank, result in enumerate(results, start=1):
        rank = int(result["rank"])
        if rank != expected_rank:
            raise ValueError("Ranks must be consecutive from 1")

        chunk_id = result["chunk_id"]
        contribution = 1.0 / (rrf_k + rank)
        candidate = candidates.setdefault(
            chunk_id,
            {
                "source_ranks": {},
                "source_contributions": {},
                "_score": 0.0,
            },
        )
        candidate["source_ranks"][source_name] = rank
        candidate["source_contributions"][source_name] = contribution
        candidate["_score"] += contribution
```

每条输出不仅保存 RRF 总分，还保留：

```json
{
  "source_ranks": {"bm25": 2, "dense": 1},
  "source_contributions": {
    "bm25": 0.016129032,
    "dense": 0.016393443
  }
}
```

这使 Hybrid 排名可以重新计算和审计，也能区分“两个来源共同支持”与“只被一个来源召回”。

### 2.6 确定性的并列规则

本实验按以下键排序：

```python
(
    -rrf_score,
    -source_count,
    best_source_rank,
    chunk_id,
)
```

含义依次为：

1. RRF 总分更高者优先；
2. 分数相同时，来源覆盖更多者优先；
3. 再比较候选在任一来源中的最好 Rank；
4. 仍相同时，按稳定 `chunk_id` 升序，保证跨运行一致。

最后一项只是工程上的确定性规则，不包含相关性判断。q01 的两个候选在前三项完全相同，因此 `chunk-000` 因 ID 较小排在第一；它恰好不含答案。这不能解释成 `chunk-000` 比 `chunk-001` 更相关，只能解释成“当前融合信号不足以区分它们”。

### 2.7 运行与自动验证

复现命令为：

```bash
PYTHONPATH=scripts python -m unittest discover -s tests -v
PYTHONPATH=scripts python scripts/run_rrf_hybrid_retrieval.py
```

新增的 5 个 RRF 测试验证了：

1. 同时被两个来源召回的候选会累加两份贡献；
2. `(Rank 1, Rank 2)` 与 `(Rank 2, Rank 1)` 精确并列；
3. 只在一个来源出现的候选只保留一份贡献；
4. 不连续 Rank 会被拒绝；
5. 同一 Chunk ID 在两个来源中的内容不一致会被拒绝。

加上 BM25 的 4 个测试，当前共有 9 个测试全部通过。实验入口位于 [`scripts/run_rrf_hybrid_retrieval.py`](../scripts/run_rrf_hybrid_retrieval.py)，生成：

- [`day2_rrf_hybrid_retrieval.jsonl`](../results/day2_rrf_hybrid_retrieval.jsonl)：Hybrid Top-5 与来源贡献；
- [`day2_rrf_hybrid_retrieval_summary.json`](../results/day2_rrf_hybrid_retrieval_summary.json)：融合参数、指标和并列问题；
- [`day2_rrf_retriever_comparison.csv`](../results/day2_rrf_retriever_comparison.csv)：BM25、Dense、Hybrid 逐问题对照。

### 2.8 运行结果

三条检索管线的答案级指标为：

| 指标 | BM25 | Dense | RRF Hybrid |
|---|---:|---:|---:|
| Gold Answer Chunk Recall@1 | 1.0 | 0.8 | 0.8 |
| Gold Answer Chunk Recall@3 | 1.0 | 1.0 | 1.0 |
| Gold Answer Chunk MRR | 1.0 | 0.9 | 0.9 |

Hybrid Top-1 与 BM25 的一致率为 0.8，与 Dense 的一致率为 0.6。逐问题结果为：

| 问题 | BM25 答案 Rank | Dense 答案 Rank | Hybrid 答案 Rank | Top-1 是否并列 |
|---|---:|---:|---:|---|
| q01 退款 | 1 | 2 | 2 | 是 |
| q02 发票 | 1 | 1 | 1 | 是 |
| q03 会员 | 1 | 1 | 1 | 否 |
| q04 保修 | 1 | 1 | 1 | 否 |
| q05 物流 | 1 | 1 | 1 | 是 |

5 个问题中有 3 个出现 Top-1 RRF 分数并列：

- q01：两个退款 Chunk 的 Rank 1/2 对称交换，只有其中一个含答案；
- q02：两个发票 Chunk 对称交换，但 overlap 使两者都含答案；
- q05：两个物流 Chunk 对称交换，overlap 同样使两者都含答案。

这解释了为什么同样的并列结构只在 q01 造成指标下降：并列本身是 Retriever 层现象，是否影响答案还取决于候选证据是否因 overlap 而完整。

### 2.9 为什么本次 Hybrid 没有超过最佳单路

RRF 常被描述为稳健的无训练融合方法，但“稳健”不等于“必然超过每个单路”。本实验中：

1. BM25 已经在 5 个问题上全部把答案 Chunk 排到第一；
2. Dense 只在 q01 把答案 Chunk 排到第二；
3. q01 中两个来源恰好给出对称 Rank，RRF 无法决定应该相信 BM25 还是 Dense；
4. 确定性 ID 并列规则选择了不含答案的 `chunk-000`。

因此 Hybrid 的 Answer Recall@1 为 0.8，与 Dense 相同，低于 BM25 的 1.0。它没有破坏 Top-3 覆盖，说明答案仍在候选集内；后续 Reranker 可以利用 Query–Chunk 联合语义再次区分这两个并列候选。

更一般地说，RRF 的价值主要体现在两个来源拥有互补召回时。当前数据过小、问题措辞又接近正文，BM25 已接近饱和，没有留下足够的互补空间让 Hybrid 提升。

### 2.10 与知识库版权保护的关系

RRF 对水印信号有三种典型作用：

- **跨 Retriever 共识**：水印 Chunk 同时在 BM25 与 Dense 排名靠前时，贡献累加，Hybrid 更可能保留它；
- **单路特异信号稀释**：水印只依赖罕见词项或只依赖向量方向时，可能只被一个 Retriever 召回，RRF 分数通常低于双路共识候选；
- **对称冲突与歧义**：正常 Chunk 和水印 Chunk 分别被两个 Retriever 偏好时，可能形成 q01 式并列，使水印是否进入 Top-k 取决于候选深度和并列规则。

这意味着设计跨检索器水印时，不能只优化某一条 Retriever 的 Rank 1。更稳健的目标应包括：

1. 在多种 Retriever 中都进入候选池；
2. 避免只靠单一词面触发或单一向量方向；
3. 报告融合前后目标水印 Chunk 的 Rank 与来源贡献；
4. 把并列策略纳入实现和威胁模型；
5. 继续测量 Reranker 是否会打破并列并过滤水印。

### 2.11 实验边界与易错点

- 输入仅为各来源 Top-5，未进入任一 Top-5 的 Chunk 不可能被 Hybrid 恢复；
- `k=60` 是常用默认值，不代表对当前语料最优；
- 未使用 Weighted RRF，默认 BM25 与 Dense 同等可信；
- 只有 5 个问题，3/5 并列率不能外推到真实知识库；
- 当前没有正常/水印查询对，尚不能计算跨 Retriever Transfer Rate；
- overlap 让 q02、q05 的两个相邻 Chunk 都含答案，掩盖了并列对端到端问答的潜在影响；
- RRF 只重排候选，不读取 Query–Chunk 内容，无法替代 Cross-Encoder Reranker。

## 3. Qwen3 Reranker 全量候选重排实验

### 3.1 知识定位：Reranker 与 Dense Retriever 的根本区别

Dense Retriever 使用双塔结构分别编码 Query 和 Chunk：

```text
Query ──Query Encoder────→ q
                              ┐
                              ├─ Inner Product / Cosine
                              ┘
Chunk ──Document Encoder─→ d
```

Chunk 向量可以离线预计算，在线只需编码 Query 并搜索向量索引。这使 Dense Retriever 适合在大规模知识库中快速召回候选，但 Query 与 Chunk 在编码阶段没有逐 Token 交互。

Reranker 则把两者放进同一个模型输入：

```text
[Instruction, Query, Chunk] → Qwen3 Reranker → relevance score
```

模型的 Self-Attention 可以直接比较 Query 中的要求和 Chunk 中的具体证据。代价是每个 Query–Chunk 对都需要单独经过模型，不能提前为所有 Query 预计算一个通用分数。因此典型系统先用 BM25/Dense 召回几十或几百个候选，再用 Reranker 做较昂贵的精排。

| 组件 | 输入方式 | 能否预计算 Chunk 表示 | 主要职责 |
|---|---|---|---|
| Dense Retriever | Query、Chunk 分别编码 | 可以 | 从全库快速召回 |
| RRF | 只读取多个来源的 Rank | 不涉及模型表示 | 融合异构 Retriever |
| Qwen3 Reranker | Query 与一个 Chunk 联合输入 | 不可以 | 在小候选集内精排 |

### 3.2 完整重排数据流

```mermaid
flowchart LR
    Q["Query"] --> BM["BM25 全量 Top-12"]
    Q --> DE["Dense 全量 Top-12"]
    BM --> RRF["RRF 候选并集<br/>12 个 Chunk"]
    DE --> RRF
    RRF --> PAIR["构造 12 个<br/>Query–Chunk Pair"]
    PAIR --> QR["Qwen3-Reranker-0.6B<br/>联合编码"]
    QR --> LOGIT["yes/no logits"]
    LOGIT --> SCORE["logit difference<br/>与 relevance probability"]
    SCORE --> SORT["按相关性重新排序"]
    SORT --> TOP5["输出 Top-5<br/>保留全量 12 个得分"]
    TOP5 --> EVAL["Answer Rank / Recall / MRR<br/>Top-1 Margin"]
```

原计划使用 Hybrid Top-30。当前受控知识库总共只有 12 个 Chunk，因此没有伪造或复制候选来凑满 30，而是把两个 Retriever 的全量 Top-12 排名交给 RRF，再对 12 个不同 Chunk 全部重排，最后输出 Top-5。这是当前语料下比“已有 Top-5 再重排”更严格的等价实验。

### 3.3 Qwen3 Reranker 如何产生分数

模型输入由三个显式字段构成：

```text
<Instruct>: Given a user question, retrieve passages that contain
            sufficient evidence to answer the question
<Query>: 青岚商城普通商品签收后多久可以申请无理由退款？
<Document>: 退款期限：普通商品自物流签收次日零时起计算……
```

外层 Chat Template 要求模型判断 Document 是否满足 Query 和 Instruction，并且答案只能是 `yes` 或 `no`。实验不调用文本生成，而是直接读取最后位置上两个 Token 的 logits：

\[
z_{\text{yes}},\quad z_{\text{no}}
\]

用于排序的原始分数是 logit difference：

\[
s(q,d)=z_{\text{yes}}-z_{\text{no}}
\]

同时记录便于理解的相关概率：

\[
p(\text{relevant}\mid q,d)
=
\frac{e^{z_{\text{yes}}}}
{e^{z_{\text{yes}}}+e^{z_{\text{no}}}}
=
\sigma\left(s(q,d)\right)
\]

因为 Sigmoid 是单调函数，按 \(s\) 或 \(p\) 排序结果相同。本实验优先分析原始 \(s\) 和候选间的 logit margin，概率只作为辅助解释。

### 3.4 核心实现：从最终 Token logits 取分

透明实现位于 [`scripts/qwen_reranker.py`](../scripts/qwen_reranker.py)。Tokenize 后，程序为每个 Query–Chunk 对添加固定 Prefix 和 Suffix，再进行左侧 Padding：

```python
input_ids = [
    prefix_tokens + pair_ids + suffix_tokens
    for pair_ids in tokenized["input_ids"]
]
model_inputs = tokenizer.pad(
    {"input_ids": input_ids},
    padding=True,
    return_attention_mask=True,
    return_tensors="pt",
)
```

模型前向传播后，只读取序列最后位置：

```python
final_logits = model(**model_inputs).logits[:, -1, :]
yes_logits = final_logits[:, yes_token_id].float()
no_logits = final_logits[:, no_token_id].float()

binary_logits = torch.stack([no_logits, yes_logits], dim=1)
probabilities = torch.softmax(binary_logits, dim=1)[:, 1]
logit_differences = yes_logits - no_logits
```

最终排序键为：

```python
(
    -reranker_logit_difference,
    hybrid_rank,
    chunk_id,
)
```

模型分数相同时才回退到 Hybrid Rank 和稳定 Chunk ID。输出同时保存重排前后的 Rank、RRF 分数、Reranker logit、概率和输入 Token 数，因此一次排序变化可以追溯到完整的前后状态。

### 3.5 模型与运行环境

| 项目 | 固定值 |
|---|---|
| 模型 | `Qwen/Qwen3-Reranker-0.6B` |
| Revision | `5340c0261aa49a842d1bff01db91ce407bda87a2` |
| 权重 SHA-256 | `27cd75a405b9c1b4…c5e65b` |
| 参数精度 | BF16 |
| 最大输入 | 8192 Token |
| 候选深度 | 全量 12 |
| 输出深度 | Top-5 |
| GPU | 单张 NVIDIA L20 |
| Transformers | 4.57.6 |
| Torch | 2.6.0+cu124 |
| Python | 3.10.19 |

模型快照只保存在 `/data/haojiachen/rag/models/huggingface`，正式实验设置 `HF_HUB_OFFLINE=1` 并成功加载。固定 Revision、权重哈希与离线验证记录见 [`day2_reranker_model_manifest.json`](../results/day2_reranker_model_manifest.json)。

### 3.6 运行与自动验证

本地不加载模型权重，只验证格式化和排序逻辑：

```bash
PYTHONPATH=scripts python -m unittest discover -s tests -v
```

新增的 4 个测试覆盖：

1. Instruction、Query、Document 字段格式；
2. 模型分数能够覆盖原 Hybrid Rank；
3. 模型分数相同时回退到 Hybrid Rank；
4. 候选数和分数数不一致时拒绝运行。

连同 BM25 与 RRF 测试，当前 13 个测试全部通过。服务器正式运行命令为：

```bash
HF_HUB_OFFLINE=1 bash scripts/run_server_python.sh \
  scripts/run_qwen_reranker.py \
  --hybrid-traces tmp/day2_rrf_hybrid_top12.jsonl \
  --candidate-depth 12 \
  --output-top-k 5
```

实验产物包括：

- [`day2_qwen_reranker.jsonl`](../results/day2_qwen_reranker.jsonl)：5 个问题的完整 12 候选分数、格式化输入和 Top-5；
- [`day2_qwen_reranker_summary.json`](../results/day2_qwen_reranker_summary.json)：指标、逐问题 Margin、耗时和环境；
- [`day2_reranker_comparison.csv`](../results/day2_reranker_comparison.csv)：Hybrid 与 Reranker 的逐问题 Rank 变化。

### 3.7 四条管线的运行结果

| 指标 | BM25 | Dense | RRF Hybrid | Qwen3 Reranker |
|---|---:|---:|---:|---:|
| Gold Answer Chunk Recall@1 | 1.0 | 0.8 | 0.8 | 1.0 |
| Gold Answer Chunk Recall@3 | 1.0 | 1.0 | 1.0 | 1.0 |
| Gold Answer Chunk MRR | 1.0 | 0.9 | 0.9 | 1.0 |

逐问题变化为：

| 问题 | Hybrid Top-1 | Reranker Top-1 | 答案 Rank 变化 | Top-1 Logit Margin |
|---|---|---|---:|---:|
| q01 退款 | `refund#chunk-000` | `refund#chunk-001` | 2 → 1 | 1.2500 |
| q02 发票 | `invoice#chunk-000` | `invoice#chunk-000` | 1 → 1 | 0.2500 |
| q03 会员 | `membership#chunk-001` | `membership#chunk-001` | 1 → 1 | 5.0625 |
| q04 保修 | `warranty#chunk-000` | `warranty#chunk-001` | 1 → 1 | 0.5625 |
| q05 物流 | `logistics#chunk-000` | `logistics#chunk-001` | 1 → 1 | 0.7500 |

5 个问题中有 3 个 Top-1 Chunk 发生变化；只有 q01 的 Gold Answer Rank 真正改善，因为 q04、q05 的相邻 Chunk 都因 overlap 包含答案。

### 3.8 重点案例：Reranker 如何打破 q01 并列

q01 的前三名为：

| Reranker Rank | Chunk | Hybrid Rank | Logit Difference | 概率 | 是否含答案 |
|---:|---|---:|---:|---:|---|
| 1 | `refund#chunk-001` | 2 | 6.8125 | 0.998901248 | 是 |
| 2 | `refund#chunk-002` | 3 | 5.5625 | 0.996175528 | 否 |
| 3 | `refund#chunk-000` | 1 | 4.3125 | 0.986777246 | 否 |

RRF 只能看到 `chunk-000` 与 `chunk-001` 分别取得 `(BM25 2, Dense 1)` 和 `(BM25 1, Dense 2)`，因此给出相同分数。Reranker 直接读取文本后发现：

- `chunk-001` 同时包含“普通商品、签收、9 个自然日、申请无理由退款”，能够完整回答；
- `chunk-000` 只包含退款适用范围和申请材料；
- `chunk-002` 提到不支持无理由退款的例外与审核流程，但没有申请期限。

于是正确证据从 Hybrid Rank 2 提升到 Rank 1，Top-1 与 Top-2 的 logit margin 为 `1.25`。

### 3.9 高概率不等于证据充分

虽然 q01 的正确 Chunk 概率为 `0.9989`，不含具体答案的 `chunk-000` 也得到 `0.9868`。这说明：

1. 当前 yes/no 概率没有在本任务上做校准；
2. 模型可能把“主题高度相关”也判为高相关；
3. Sigmoid 在大正 Logit 区域容易饱和，概率差看起来很小；
4. Reranker 的主要用途是候选间相对排序，不应仅凭固定概率阈值判断“证据充分”。

因此实验同时保存原始 logit difference 和 Top-1 margin。若后续把 Reranker 作为版权 Detector 或过滤器，还需要独立校准集、FPR、ROC/PR 曲线和阈值置信区间。

### 3.10 运行成本

| 项目 | 结果 |
|---|---:|
| 模型加载 | 1.079 秒 |
| 5 个问题、共 60 对候选 | 0.676 秒 |
| 首个 12 对 Batch | 0.464 秒 |
| 预热后每个 12 对 Batch | 约 0.045 秒 |
| 峰值 GPU 显存 | 1.959 GiB |

首个 Batch 包含 CUDA Kernel 初始化等预热成本，因此明显慢于后续 Batch。当前语料很短，不能将这些吞吐数字直接外推到 8K 或 32K 长文本；Reranker 成本会随候选数和序列长度增长。

### 3.11 与知识库版权保护的关系

Reranker 是水印从“被召回”到“进入 Prompt”之间的第二道门：

- 只利用罕见词面提高 BM25 Rank 的水印，若与 Query 的回答关系弱，可能被 Reranker 降级；
- 只优化 Dense 向量接近性的水印，也可能因联合语义不充分而被过滤；
- 同时保持自然语义、问题相关性和目标知识的水印，更可能在重排后存活；
- Reranker Instruction、模型版本、候选深度和输入截断都可能改变水印 Rank；
- 未经校准的高相关概率不能直接作为所有权信号，否则容易产生误触发和 Spoofing。

因此后续水印实验必须同时记录：

```text
目标水印 Chunk 的 Retriever Rank
→ RRF Rank
→ Reranker Rank 与 Margin
→ 是否进入最终 Context
→ 输出或 Detector 是否命中
```

一个只报告 Dense Top-k 命中的水印方案，尚不能证明它能穿过现代 RAG 的 Reranker。

### 3.12 实验边界与易错点

- 当前只有 5 个问题和 12 个 Chunk，指标仍是机制验证；
- Top-30 被适配为当前语料的全量 Top-12，尚未测量真实大候选池；
- 问题措辞接近正文，不能代表同义改写和跨语言查询；
- 每个条件只运行一次，未测量不同硬件和数值精度下的微小分数变化；
- 概率未校准，不能直接充当证据充分性阈值；
- q04、q05 的答案存在于两个 overlap Chunk，Top-1 改变没有影响答案 Recall；
- 尚未加入正常/水印查询对，不能报告 Reranker 对水印的保留率和误触发率。

## 4. 事实复制型 Canary 正对照与设计纠错

### 4.1 实验定位：这是正对照，不能验证水印因果效果

首轮实验最初试图回答：

> 一个同时出现在 Query 和目标 Chunk 中的唯一触发短语，能否稳定穿过 BM25、Dense、RRF 和 Reranker？

复核后发现，这个设计不能回答该问题：水印 Chunk 直接复制了普通查询的 Gold 事实，因此即使删除触发词，它仍然是高度相关文档。实验测到的高 Hit@k 同时包含“事实本身相关”和“触发词匹配”两种效应，无法归因。

因此本节保留其结果，只把它解释为事实复制型正对照，用于展示“相关 Canary 在四条管线中如何排序”；这些数字不再作为水印有效性、误触发率或迁移率的正式结论，也不作为 `plan.md` 水印检索任务的完成依据。

在原有 12 个干净 Chunk 上追加 20 个水印 Chunk。每个水印 Chunk 包含：

```text
版本追踪标记：<唯一触发短语>。规则摘要：<一个真实政策事实>
```

每个目标 Chunk 配一对语义相同的查询：

```text
正常查询：青岚智能设备主体的有限保修期是多久？
水印查询：橙枫轴距。青岚智能设备主体的有限保修期是多久？
```

触发词“橙枫轴距”只出现在该水印查询和其目标 Chunk 中。20 个触发词均不出现在原始干净语料，且互不重复。数据由 [`build_watermark_retrieval_dataset.py`](../scripts/build_watermark_retrieval_dataset.py) 生成并自动验证。

这种成对设计保持问题语义不变，因此两次排名的差异主要来自触发短语；但水印 Chunk 同时复制了真实事实，所以正常查询也可能把它当作相关证据召回。这一点会直接表现为误触发，而不是被实验设计隐藏。

### 4.2 完整数据流

```mermaid
flowchart LR
    C["12 个干净 Chunk"] --> INDEX["追加 20 个<br/>水印目标 Chunk"]
    W["20 对 Query"] --> NQ["Normal Query<br/>不含触发词"]
    W --> TQ["Watermarked Query<br/>含唯一触发词"]
    INDEX --> BM["BM25 全库排名"]
    INDEX --> DE["Dense 全库排名"]
    NQ --> BM
    NQ --> DE
    TQ --> BM
    TQ --> DE
    BM --> RRF["RRF 全库融合"]
    DE --> RRF
    RRF --> RR["Qwen3 Reranker<br/>全量 32 候选"]
    BM --> METRIC["Target Rank / Hit@k<br/>Margin / False Trigger"]
    DE --> METRIC
    RRF --> METRIC
    RR --> METRIC
    METRIC --> TRANS["跨 Retriever<br/>条件迁移率"]
```

共运行 40 条查询。每条查询在四条管线中都保留 32 个 Chunk 的完整排名，Reranker 总计打分 \(40\times32=1280\) 个 Query–Chunk 对。

### 4.3 指标：分别衡量触发能力和触发专属性

设第 \(i\) 对查询的目标水印 Chunk 为 \(d_i^w\)，它在检索器 \(R\) 下的名次为 \(r_R(q,d_i^w)\)。

目标命中率衡量水印查询能否召回指定 Chunk：

\[
\operatorname{Hit@k}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}\left[
r_R(q_i^w,d_i^w)\le k
\right]
\]

触发带来的名次提升定义为：

\[
\Delta r_i
=
r_R(q_i^{normal},d_i^w)
-
r_R(q_i^w,d_i^w)
\]

\(\Delta r_i>0\) 表示加入触发词后目标 Chunk 上升。实验还记录目标与下一名的分数间隔：

\[
\operatorname{Margin}_{next}
=
s(q,d_{r})-s(q,d_{r+1})
\]

不同检索器的分数不在同一量纲，因此 Margin 只能在同一检索器内部解释，不能直接比较 BM25 的 `23.97` 和 Dense 的 `0.11`。

正常查询误触发同时采用两种定义：

- `Exact-target FTR@k`：正常查询是否召回与它配对的目标水印 Chunk；
- `Any-watermark Exposure@k`：正常查询的 Top-k 中是否出现任意水印 Chunk。

前者衡量触发专属性，后者衡量水印内容对正常候选集合的污染。跨检索器迁移率采用条件定义：

\[
\operatorname{Transfer}_{A\rightarrow B}@k
=
\frac{
\left|H_A^k\cap H_B^k\right|
}{
\left|H_A^k\right|
}
\]

其中 \(H_A^k\) 是在检索器 \(A\) 上成功进入 Top-k 的水印集合。

### 4.4 核心实现

数据构造时显式检查触发词唯一性：

```python
if any(trigger in clean_text for trigger in triggers):
    raise AssertionError(
        "A watermark trigger already occurs in the clean corpus"
    )

matching_chunks = [
    chunk["chunk_id"]
    for chunk in watermark_chunks
    if trigger in chunk["text"]
]
if matching_chunks != [pair["target_chunk_id"]]:
    raise AssertionError(
        f"Trigger is not unique to its target: {trigger}"
    )
```

四条管线使用同一批全量候选：

```python
bm25_trace = bm25.search(query, top_k=len(chunks))
dense_trace = dense.search(query, top_k=len(chunks))

hybrid_trace = reciprocal_rank_fusion(
    {
        "bm25": bm25_trace["results"],
        "dense": dense_trace["results"],
    },
    rrf_k=60,
    top_k=len(chunks),
)

reranker_trace = reranker.rerank(
    query,
    hybrid_trace["results"],
)
```

目标 Rank、下一名 Margin 和水印暴露由统一函数计算：

```python
target_index = next(
    (
        index
        for index, result in enumerate(results)
        if result["chunk_id"] == target_chunk_id
    ),
    None,
)
rank = None if target_index is None else target_index + 1

target_gap_to_next = (
    results[target_index]["score"]
    - results[target_index + 1]["score"]
)
```

完整实验入口为 [`run_watermark_retrieval_experiment.py`](../scripts/run_watermark_retrieval_experiment.py)，模型无关指标位于 [`watermark_retrieval_metrics.py`](../scripts/watermark_retrieval_metrics.py)。

### 4.5 运行设置与验证

| 项目 | 设置 |
|---|---|
| 干净/水印 Chunk | 12 / 20 |
| 查询对/查询总数 | 20 / 40 |
| BM25 | \(k_1=1.5,b=0.75\) |
| Dense | 固定 Revision 的 `Qwen3-Embedding-0.6B` |
| Dense Index | 归一化向量 + `FAISS IndexFlatIP` |
| RRF | \(k=60\)，两路均输入全量 Top-32 |
| Reranker | 固定 Revision 的 `Qwen3-Reranker-0.6B` |
| Reranker 候选 | 全量 32 |
| GPU | 单张 NVIDIA L20 |

本地 17 项测试全部通过；服务器新增数据与指标测试 4/4 通过。正式运行使用离线模型缓存，耗时如下：

| 阶段 | 时间 |
|---|---:|
| Dense 模型加载 | 1.153 秒 |
| 32 个 Chunk 编码与建库 | 0.343 秒 |
| Reranker 模型加载 | 0.681 秒 |
| 1,280 对 Reranker 打分 | 4.836 秒 |
| 实验总耗时 | 8.446 秒 |

### 4.6 运行结果：仅适用于事实复制型正对照

水印查询和正常查询的核心结果如下：

| 管线 | 水印 Hit@1 | 水印 Hit@5 | 正常 Exact FTR@1 | 正常 Exact FTR@5 | 平均 Rank Gain |
|---|---:|---:|---:|---:|---:|
| BM25 | 1.00 | 1.00 | 0.50 | 1.00 | 0.75 |
| Dense | 1.00 | 1.00 | 0.05 | 0.85 | 2.55 |
| RRF Hybrid | 1.00 | 1.00 | 0.35 | 1.00 | 1.25 |
| Qwen3 Reranker | 0.90 | 1.00 | 0.20 | 0.95 | 1.10 |

四条管线的水印查询 Hit@5 都为 1.0，说明该显式触发信号可以稳定进入较宽候选集。BM25、Dense 和 RRF 的 Hit@1 也都是 1.0；Reranker 的 Hit@1 降为 0.9，即 20 个目标中有 2 个被降到 Rank 2。

Dense 的平均 Rank Gain 最大，为 `2.55`，并且正常查询 Exact FTR@1 只有 `0.05`。这意味着在当前数据上，唯一短语不仅能进入 Dense 语义表示，还比 BM25 更明显地区分“有触发”和“无触发”的成对查询。不过到 Top-5 时 Dense 的正常 Exact FTR 已升到 `0.85`，说明目标 Chunk 本身复制了正确事实，即使没有触发词也经常被召回。

BM25 的水印目标相对下一名平均 Margin 为 `23.967676`，触发短语的罕见 unigram/bigram 带来很强的精确匹配贡献。但正常查询 Exact FTR@1 已达到 `0.50`，表明政策事实的字面重合本身就足以让一半水印副本排到第一。BM25 在这里“触发强”，却不等于“触发专一”。

四条管线在 Top-5 的 `Any-watermark Exposure` 都为 1.0。原因不是所有触发词都会被正常查询猜中，而是水印 Chunk 占语料的 `20/32`，并复制了正常问题所需事实。这一结果说明：在评估误触发时，不能只看任意水印文档是否进入候选；还必须报告成对目标误触发，并在后续实验减少事实复制和水印占比。

### 4.7 Reranker 为什么过滤了两个水印 Top-1

`wm18` 查询为：

```text
岩鲸序列。青岚包裹多久没有新增物流轨迹会被标记为异常停滞？
```

Reranker 的前两名是：

| Rank | Chunk | Logit Difference |
|---:|---|---:|
| 1 | 原始物流政策 `chunk-001` | 8.7500 |
| 2 | `canary-wm18#chunk-000` | 8.3750 |

`wm19` 也出现相同现象：

| Rank | Chunk | Logit Difference |
|---:|---|---:|
| 1 | 原始物流政策 `chunk-001` | 7.8750 |
| 2 | `canary-wm19#chunk-000` | 7.4375 |

两个 Canary 都包含正确答案和唯一触发词，但原始政策 Chunk 同时包含停滞判定、计时起点、核查流程和补偿规则。Reranker 的 Instruction 要求寻找“包含充分回答证据的段落”，因此更完整的原始 Chunk 得分更高。

这说明：

```text
唯一触发词
→ BM25/Dense/RRF Rank 1
→ Reranker 联合检查回答充分性
→ 2 个短 Canary 被完整原文压到 Rank 2
```

Reranker 没有完全删除水印：两例仍在 Top-5。但它证明了纯词面或向量触发不能保证最终 Rank 1，水印 Chunk 还必须在语义上符合重排任务。

### 4.8 表观迁移与安全含义

BM25、Dense 和 RRF 的水印 Hit@1 都是 20/20，因此从任一条管线迁移到另外两条的条件迁移率均为 1.0；迁移到 Reranker 的 Hit@1 为 0.9。Top-5 时四条管线间迁移率全部为 1.0。

当前正对照只支持以下机制性结论：

1. 显式唯一短语与完整答案同时存在时，可以在 BM25、Dense 和 RRF 中形成很强的表观召回，但不能确定贡献来自哪一部分；
2. RRF 会保留两个来源都支持的目标，但不会检查水印是否真的提供充分证据；
3. Reranker 是第一道显式检查 Query–Chunk 回答关系的组件，会削弱“只靠触发短语取得 Rank 1”的水印；
4. 高 Hit@k 不能单独证明水印设计良好；当前正常查询 FTR 很高，说明触发专属性和无害性不足；
5. 后续水印设计应减少事实复制，控制水印文档占比，并把 Reranker 后的 Rank 与 Margin 纳入优化目标。

### 4.9 实验边界

- 语料仅有 32 个 Chunk，Top-20 很宽，Hit@20 没有足够区分度；
- 水印 Chunk 占比高达 62.5%，会放大正常查询的水印暴露；
- 每个 Canary 直接复制一个真实事实，因此正常查询召回它并不完全等价于面向最终 Detector 的假阳性；
- 触发词均为显式罕见中文四字短语，尚未测试同义改写、字符扰动和隐蔽短语；
- 当前只测检索到目标 Chunk，没有运行 Generator 或所有权 Detector；
- Reranker 全量处理 32 个候选，尚未模拟真实系统中目标在候选截断前就丢失的情况；
- Margin 只能在各检索器内部比较，不能跨 BM25、Dense、RRF 和 Reranker 直接比较绝对值。

### 4.10 纠正后的三条件设计

纠正版不再把业务答案复制进 Canary。每个样本显式保存正常问题对应的 Clean Gold Chunk，并建立三个查询条件：

```text
Normal
  普通业务问题，不含触发词

Trigger-only control
  唯一触发词 + 同一个普通业务问题
  Canary 含触发词，但不能回答该业务问题

Semantic verification
  询问该触发标记对应的核验口令
  Canary 能够直接、充分回答
```

纠正后的 Canary 形式为：

```text
版权核验记录：版本追踪标记“岩鲸序列”的核验口令为
“QL-AUTH-18”。该记录仅用于知识库授权副本验证，
不描述商城业务规则。
```

对应三个查询为：

```text
Normal:
青岚包裹多久没有新增物流轨迹会被标记为异常停滞？

Trigger-only:
岩鲸序列。青岚包裹多久没有新增物流轨迹会被标记为异常停滞？

Semantic verification:
青岚知识库中，版本追踪标记“岩鲸序列”的核验口令是什么？
```

三个条件分别回答不同问题：

| 对照 | 想测量的效应 | 理想结果 |
|---|---|---|
| Normal | 普通查询误暴露 | Canary Rank 低 |
| Trigger-only | 纯词面/向量触发 | BM25/Dense Rank 上升，Reranker 可能过滤 |
| Semantic verification | Reranker-aware 验证 | 四条管线都保留目标 Canary |

形式化地，纯触发效应要求：

\[
r(q_{\text{normal}},d_w)
>
r(q_{\text{trigger-only}},d_w)
\]

而端到端验证要求：

\[
r(q_{\text{verification}},d_w)\le k
\]

同时保持 Normal 条件的误暴露较低。修正版还新增以下自动检查：

- Canary 不包含原业务问题的 `source_fact`；
- Canary 包含验证查询所需的唯一核验口令；
- 每条 Normal Query 显式关联一个或多个 Clean Gold Chunk；
- 触发词只出现在自己的 Canary 与两个触发条件查询中；
- 三种条件使用同一组 32 个候选，避免语料变化造成混淆。

修正版数据生成与 17 项本地测试全部通过。随后在单张 L20 上使用固定 Revision 的 Embedding 和 Reranker，离线运行 60 条查询、四条管线和全量 32 候选，共保存 240 条完整 Trace。

### 4.11 修正版运行结果

三个条件的核心指标如下：

| 管线 | Normal Exact FTR@1 | Trigger-only Hit@1 | Trigger-only Hit@5 | Verification Hit@1 | Trigger-only 平均 Rank Gain |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.00 | 0.00 | 1.00 | 1.00 | 17.95 |
| Dense | 0.00 | 0.15 | 1.00 | 1.00 | 13.95 |
| RRF Hybrid | 0.00 | 0.10 | 1.00 | 1.00 | 16.90 |
| Qwen3 Reranker | 0.00 | 0.00 | 0.95 | 1.00 | 16.65 |

这组结果把三种效应清楚分开：

1. **Normal**：四条管线的 Exact FTR@1 都是 0，说明 Canary 不再因为复制业务答案而直接排到第一；
2. **Trigger-only**：仅加入唯一触发词后，BM25、Dense 和 RRF 的目标 Hit@5 都从低基线升到 1.0，证明触发词确实改变了检索 Rank；
3. **Semantic verification**：专用验证查询在四条管线中的 Hit@1 都是 1.0，说明当 Canary 能充分回答 Query 时，Reranker 会完整保留水印。

Normal 条件的 Exact FTR@5 分别为：

| BM25 | Dense | RRF | Reranker |
|---:|---:|---:|---:|
| 0.00 | 0.05 | 0.05 | 0.00 |

这比首轮事实复制正对照的 `1.00/0.85/1.00/0.95` 显著降低。需要区分 Exact FTR 与 Any-watermark Exposure：Dense 的 Normal Any-watermark Exposure@5 仍为 0.65，因为 32 个 Chunk 中有 20 个格式相似的 Canary；但正常查询命中与自己配对的特定目标仅为 0.05。后续扩大干净语料并降低 Canary 占比后，才适合估计面向真实知识库的绝对暴露率。

目标相对下一名的平均 Margin 为：

| 条件 | BM25 | Dense | RRF | Reranker |
|---|---:|---:|---:|---:|
| Trigger-only | 7.252242 | 0.071733 | 0.000570 | 2.028125 |
| Verification | 21.799434 | 0.155020 | 0.000769 | 6.675000 |

这些数值只能在同一检索器内部比较。四条管线中，Verification Margin 都高于 Trigger-only，说明“Query–Chunk 具有可回答关系”比“只共享罕见触发词”形成了更稳定的目标分离。

### 4.12 Reranker 过滤案例：wm03

`wm03` 的 Trigger-only Query 是：

```text
银杉回声。青岚商城普通商品的无理由退款申请期限是多久？
```

其 Canary 只说明：

```text
版本追踪标记“银杉回声”的核验口令为“QL-AUTH-03”。
```

它不包含“9 个自然日”的退款答案。四条管线中的目标 Rank 为：

| BM25 | Dense | RRF Hybrid | Qwen3 Reranker |
|---:|---:|---:|---:|
| 4 | 3 | 3 | 7 |

Reranker Rank 1 是真正包含答案的 `qinglan-refund-v1#chunk-001`，其 Logit Difference 为 `5.4375`；Canary 的 Logit Difference 为 `-2.3750`。因此目标从 Hybrid Rank 3 降至 Reranker Rank 7，成为 Trigger-only 条件中唯一掉出 Top-5 的样本。

```text
罕见触发词匹配
→ Retriever 将 Canary 召回到 Top-5
→ Reranker 检查回答充分性
→ Canary 不含退款期限
→ Gold Chunk Rank 1，Canary Rank 7
```

在 Trigger-only Top-5 上，从 BM25、Dense 或 RRF 成功集合迁移到 Reranker 的条件迁移率均为 `0.95`；Verification 条件的四路 Hit@1 迁移率全部为 `1.0`。这正是修正版实验要分离的现象：纯触发可以穿过 Retriever，却不保证穿过 Reranker；语义充分的验证查询可以稳定穿过完整检索管线。

正式实验总耗时 `11.712` 秒，其中 1,920 个 Query–Chunk Reranker 打分耗时 `7.447` 秒。完整结果见 [`day2_watermark_retrieval_summary.json`](../results/day2_watermark_retrieval_summary.json)、[`day2_watermark_retrieval_comparison.csv`](../results/day2_watermark_retrieval_comparison.csv) 与完整 Trace。

### 4.13 修正版实验边界

- 20 个 Canary 占 32 个 Chunk 的 62.5%，Any-watermark Exposure 不能外推到真实知识库；
- 触发词和 `QL-AUTH-*` 核验口令都很显式，尚未测量隐蔽性、可猜测性和 Spoofing；
- Verification Query 与 Canary 使用高度一致的模板，Hit@1=1.0 是机制基线，不代表自然语言改写鲁棒性；
- Reranker 接收全量 32 个候选，尚未模拟目标在有限候选深度处被截断；
- 当前只验证检索到正确 Canary，没有调用 Generator 输出核验口令，也没有计算最终所有权 Detector 的 FPR 和统计功效；
- 需要在更低 Canary 占比、更大干净语料、查询改写和切分消融下重新测量。

## 5. FAISS Flat、HNSW、IVF 对照实验

### 5.1 知识定位：Embedding 模型与 ANN 索引不是同一组件

Qwen3-Embedding 决定“文本如何变成向量”，FAISS 索引决定“如何在已有向量中寻找近邻”。替换 Flat、HNSW 或 IVF 不会重新理解文本，只会改变候选搜索方式：

```text
Query 文本
→ Qwen3-Embedding
→ 同一个 1024 维归一化向量
→ Flat / HNSW / IVF
→ 不同速度与近似误差的 Top-k
```

三种索引的核心差异如下：

| 索引 | 搜索方式 | 是否需要训练 | 主要代价 | 是否精确 |
|---|---|---|---|---|
| Flat | 与全部 \(N\) 个向量计算内积 | 否 | 搜索量约 \(O(Nd)\) | 是 |
| HNSW | 在多层近邻图上逐步导航 | 否 | 建图慢、额外保存图边 | 否 |
| IVF | 先选择若干聚类桶，再扫描桶内向量 | 是，需要训练质心 | 训练与参数选择 | 否 |

Flat 是本实验的 Gold 邻居来源。ANN 结果必须与 Flat 比较，不能用 HNSW 或 IVF 自己的结果评价自己。

### 5.2 三种索引的数据流

```mermaid
flowchart TB
    D["归一化 Document Embeddings"] --> FLAT["Flat<br/>保存全部向量"]
    D --> HBUILD["HNSW 建图<br/>M / efConstruction"]
    D --> TRAIN["IVF 训练质心<br/>nlist"]
    TRAIN --> ASSIGN["向量分配到倒排桶"]

    Q["归一化 Query Embedding"] --> FS["全量内积扫描"]
    Q --> HS["图导航<br/>efSearch"]
    Q --> IS["选择聚类桶<br/>nprobe"]

    FLAT --> FS
    HBUILD --> HS
    ASSIGN --> IS

    FS --> GOLD["精确 Top-k"]
    HS --> ANN1["近似 Top-k"]
    IS --> ANN2["近似 Top-k"]
    GOLD --> EVAL["Recall@k / Top-1 Agreement"]
    ANN1 --> EVAL
    ANN2 --> EVAL
```

HNSW 的 `efSearch` 表示搜索时保留和探索的候选规模；值越大，通常召回越高、延迟越大。IVF 的 `nprobe` 表示查询多少个聚类桶；当 `nprobe=nlist` 时会探测所有桶，逐渐退化为接近全量扫描。

### 5.3 为什么需要真实语料和压力集两层实验

当前只有 32 个真实 Chunk。对如此小的索引：

- Flat 已经非常便宜；
- HNSW 很容易搜索完整；
- IVF 没有足够训练样本学习可靠质心；
- 微秒级延迟主要受计时噪声影响。

因此实验分成两层：

1. **真实 32 向量**：只验证 Top-10、Top-1 和 Verification Canary Rank 是否一致；
2. **8,192 向量压力集**：在 32 个真实 Document Embedding 周围生成归一化高斯邻域，用于观察参数变化趋势。

压力集生成公式为：

\[
\tilde d_{i,j}
=
\operatorname{normalize}
\left(
d_i+\epsilon_{i,j}
\right),
\qquad
\epsilon_{i,j}\sim\mathcal N(0,0.02^2I)
\]

每个真实向量生成 256 个邻域向量，固定随机种子 `20260725`。这种构造保留了真实 Embedding 的局部方向，但人为形成了 32 个清晰簇，因此天然有利于 IVF；结果只能解释机制，不能宣称 IVF 普遍优于 HNSW。

### 5.4 评价指标

对每条查询，Flat 返回精确集合 \(G_q^k\)，ANN 返回集合 \(A_q^k\)。Recall@k 为：

\[
\operatorname{Recall@k}
=
\frac{1}{|Q|}
\sum_{q\in Q}
\frac{|G_q^k\cap A_q^k|}{k}
\]

Top-1 Agreement 只检查 ANN 第一名是否等于 Flat 第一名：

\[
\operatorname{Top1Agree}
=
\frac{1}{|Q|}
\sum_{q\in Q}
\mathbb 1[A_q^1=G_q^1]
\]

这两个指标不可互相替代。例如 HNSW 可能保持正确 Top-1，却遗漏不少第 2～10 名候选；这会影响 Reranker 的候选多样性和水印 Chunk 是否进入候选池。

### 5.5 核心代码

Flat 直接保存并扫描全部向量：

```python
flat = faiss.IndexFlatIP(dimension)
flat.add(vectors)
scores, exact_ids = flat.search(queries, top_k)
```

HNSW 不需要训练，但需要建图：

```python
hnsw = faiss.IndexHNSWFlat(
    dimension,
    16,
    faiss.METRIC_INNER_PRODUCT,
)
hnsw.hnsw.efConstruction = 100
hnsw.add(vectors)
hnsw.hnsw.efSearch = 32
scores, ids = hnsw.search(queries, top_k)
```

`M=16` 控制每个节点的图连接规模，`efConstruction=100` 控制建图时的搜索深度，`efSearch` 是在线搜索变量。

IVF 必须先训练质心：

```python
quantizer = faiss.IndexFlatIP(dimension)
ivf = faiss.IndexIVFFlat(
    quantizer,
    dimension,
    64,
    faiss.METRIC_INNER_PRODUCT,
)
ivf.train(vectors)
ivf.add(vectors)
ivf.nprobe = 4
scores, ids = ivf.search(queries, top_k)
```

完整入口为 [`run_faiss_ann_comparison.py`](../scripts/run_faiss_ann_comparison.py)。实验固定 FAISS 单线程，并对 60 条查询重复搜索 30 次；延迟是每条 Query 的平均 CPU 搜索时间，不包含 Qwen Embedding 时间。

### 5.6 真实 32 Chunk 结果

在真实 32 个 Chunk 和 60 条三条件查询上：

| 索引 | Top-10 Recall vs Flat | Top-1 Agreement | Verification Hit@1 |
|---|---:|---:|---:|
| Flat | 1.0 | 1.0 | 1.0 |
| HNSW，充分搜索 | 1.0 | 1.0 | 1.0 |
| IVF，探测全部桶 | 1.0 | 1.0 | 1.0 |

所以在当前真实小语料上，替换索引不会改变 20 个 Verification Canary 的 Rank 1。

IVF 在训练 4 个质心时发出警告：32 个训练向量少于推荐训练量。这个警告比表面上的 Recall=1.0 更重要：当 `nprobe` 覆盖全部桶时可以找回精确结果，但 32 个点不足以评价 IVF 聚类质量。当前生产基线继续使用 `IndexFlatIP` 更合理。

### 5.7 8,192 向量压力集结果

| 索引配置 | Recall@10 | Top-1 一致率 | 延迟 ms/Query | QPS |
|---|---:|---:|---:|---:|
| Flat | 1.0000 | 1.0000 | 1.2123 | 824.9 |
| HNSW `efSearch=8` | 0.6350 | 0.9500 | 0.0249 | 40,126.7 |
| HNSW `efSearch=32` | 0.7100 | 0.9500 | 0.0451 | 22,150.1 |
| HNSW `efSearch=128` | 0.7683 | 1.0000 | 0.1133 | 8,826.5 |
| IVF `nprobe=1` | 0.7233 | 0.8667 | 0.0337 | 29,717.8 |
| IVF `nprobe=4` | 0.9767 | 1.0000 | 0.1077 | 9,286.1 |
| IVF `nprobe=16` | 1.0000 | 1.0000 | 0.4044 | 2,472.7 |
| IVF `nprobe=64` | 1.0000 | 1.0000 | 1.2396 | 806.7 |

结果呈现了标准的 ANN 权衡：

- HNSW 将 `efSearch` 从 8 增到 128，Recall@10 从 0.6350 升到 0.7683，同时延迟约增加 4.5 倍；
- IVF 将 `nprobe` 从 1 增到 4，Recall@10 从 0.7233 升到 0.9767，Top-1 也恢复为 1.0；
- `nprobe=16` 已在当前压力集达到精确 Top-10，但仍比 Flat 快约 3 倍；
- `nprobe=64=nlist` 时遍历全部桶，延迟 `1.2396 ms`，已经与 Flat 的 `1.2123 ms` 接近。

IVF 在这里明显优于 HNSW，不应外推为算法排名。压力集由 32 个高斯邻域组成，与 IVF 的聚类假设高度匹配；HNSW 还只测试了固定 `M=16` 和 `efConstruction=100`。真实语料的密度、维度分布、插入顺序和索引参数都会改变结果。

### 5.8 建库成本与索引体积

| 索引 | 训练时间 | 建库/添加时间 | 索引体积 |
|---|---:|---:|---:|
| Flat | 0 | 0.0187 秒 | 32.000 MiB |
| HNSW | 0 | 1.1074 秒 | 33.125 MiB |
| IVF | 0.2852 秒 | 0.0689 秒 | 32.313 MiB |

Flat 几乎没有建库结构成本；HNSW 建图最慢，并增加图边存储；IVF 需要额外训练阶段，但添加向量比 HNSW 建图便宜。这里只使用 `IVFFlat`，桶内仍保存完整 Float32 向量；若使用 PQ 压缩，体积和精度还会发生另一层权衡。

### 5.9 与知识库水印的关系

ANN 近似误差位于 Dense Retriever 内部：

```text
水印 Query 与目标 Chunk 向量接近
→ ANN 是否把目标找出来
→ 是否进入 RRF / Reranker 候选
→ 是否进入最终 Context
```

即使水印在 Flat 上 Rank 很高，也可能因 HNSW/IVF 的近似搜索而掉出候选。弱水印尤其依赖第 2～10 名候选，不能只报告 Top-1 Agreement。水印研究至少应固定并记录：

- FAISS 索引类型和 Metric；
- HNSW 的 `M`、`efConstruction`、`efSearch`；
- IVF 的 `nlist`、训练样本和 `nprobe`；
- 相对 Flat 的 Recall@k；
- 水印目标在 ANN 前后的 Rank 与 Hit@k；
- 候选深度和 Reranker 是否还有机会恢复目标。

当前 20 个 Verification Canary 在真实 32 向量上信号很强，三种索引都保持 Rank 1；这不代表更隐蔽、更弱的水印也会稳定。

### 5.10 易错点与实验边界

- 不要把 Qwen Embedding 模型与 FAISS 索引混为一谈；
- 使用 Inner Product 近似 Cosine 前必须归一化 Query 和 Document；
- IVF 训练数据必须足够且与实际语料同分布；
- 小语料上的微秒延迟和 Recall=1.0 不具有规模代表性；
- ANN Recall 必须以同向量、同 Metric 的 Flat 结果为 Gold；
- `nprobe=nlist` 失去 IVF 的主要加速意义；
- 增大 `efSearch` 或 `nprobe` 通常提升召回，但不是免费的；
- 本压力集是人为的 32 簇结构，参数结论不能直接移植到真实百万级知识库。

实验明细见 [`day2_faiss_ann_comparison.csv`](../results/day2_faiss_ann_comparison.csv)，完整参数、数据哈希和真实语料验证见 [`day2_faiss_ann_summary.json`](../results/day2_faiss_ann_summary.json)。

## 小结

本阶段先完成了一个不依赖外部检索库的透明 BM25，并在与 Dense 完全相同的 12 个 Chunk 和 5 个问题上完成对照。BM25 的 Gold Answer Chunk Recall@1 为 1.0，修复了 Dense 在 q01 上“主题正确但证据不完整”的排名错误；两种检索器的 Top-1 Chunk 一致率只有 0.4，证明它们使用的相关性信号确实不同。

随后实现的 RRF 不比较异构原始分数，只累加 BM25 与 Dense 的名次贡献。Hybrid 的 Gold Answer Chunk Recall@1 为 0.8，没有超过 BM25；q01、q02、q05 出现对称 Rank 导致的精确并列，其中 q01 因确定性 ID 规则把不含答案的 Chunk 放在第一。这个结果说明融合本身不会创造新的相关性信息：RRF 可以奖励跨 Retriever 共识，却无法判断对称冲突中哪个来源更可靠。

最后，Qwen3 Reranker 对全量 12 个 Hybrid 候选进行 Query–Chunk 联合打分，将 q01 正确证据从 Rank 2 提升到 Rank 1，使 Answer Recall@1 和 MRR 都恢复为 1.0。它还改变了 q04、q05 的 Top-1，但由于 overlap，两题的答案指标不变。高概率在同主题错误 Chunk 上同样饱和，说明 Reranker 应主要用于相对排序，而不能未经校准就充当证据充分性 Detector。

目前已经建立四条可审计管线：Dense 通过连续语义空间排序，BM25 通过 TF、IDF 和长度归一化排序，RRF 通过来源名次与共识融合，Reranker 通过 Query–Chunk 联合交互进行精排。首轮事实复制正对照证明了相关副本容易获得高召回，却不能证明水印触发有效。

纠正后的三条件实验给出了可归因结论：Normal Exact FTR@1 在四路均为 0；Trigger-only 在 BM25、Dense 和 RRF 中均达到 Hit@5=1.0，但经过 Reranker 后降为 0.95；Semantic verification 在四路中均达到 Hit@1=1.0。由此可以分别观察 Retriever 的触发敏感性、Reranker 的证据过滤，以及语义充分水印的端到端迁移。

FAISS 对照进一步把 Dense 模型与索引搜索分开：真实 32 Chunk 上 Flat、充分搜索的 HNSW 和全桶 IVF 都保持 Verification Hit@1=1.0；8,192 向量压力集上，增大 `efSearch` 或 `nprobe` 会以更高延迟换取更高 Recall。IVF 在人为 32 簇数据上表现较好是实验构造造成的，不能直接外推到真实知识库。

下一阶段可以在这一有效基线上继续水印位置、Chunk Size、Overlap 和 PCA/UMAP 消融。

## 参考资料

- [透明 Dense RAG：从文档切分到证据约束生成](./03-transparent-dense-rag.md)
- [BM25 检索器实现](../scripts/bm25_retriever.py)
- [BM25/Dense 对照实验入口](../scripts/run_bm25_retrieval.py)
- [RRF 融合实现](../scripts/rrf_fusion.py)
- [RRF Hybrid 实验入口](../scripts/run_rrf_hybrid_retrieval.py)
- [Qwen3 Reranker 实现](../scripts/qwen_reranker.py)
- [Qwen3 Reranker 实验入口](../scripts/run_qwen_reranker.py)
- [三条件水印检索数据生成器](../scripts/build_watermark_retrieval_dataset.py)
- [三条件水印查询数据](../data/eval/day2_watermark_query_triplets.jsonl)
- [水印检索统一实验入口](../scripts/run_watermark_retrieval_experiment.py)
- [水印检索指标实现](../scripts/watermark_retrieval_metrics.py)
- [水印检索实验汇总](../results/day2_watermark_retrieval_summary.json)
- [FAISS ANN 对照实验入口](../scripts/run_faiss_ann_comparison.py)
- [FAISS ANN 对照结果](../results/day2_faiss_ann_summary.json)
- [Qwen3-Reranker-0.6B 模型卡](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- Robertson, S. E. and Zaragoza, H. *The Probabilistic Relevance Framework: BM25 and Beyond*. 2009.
