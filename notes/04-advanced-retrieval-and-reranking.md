# 先进检索与重排：从 BM25 到 Hybrid RAG

本笔记围绕同一批 Chunk 和问题，逐步实现并比较 Sparse、Dense、Hybrid 与 Reranked Retrieval。每完成一个可复现实验，就增加一个教程章节；当前首先建立透明 BM25 基线，解释词项如何进入分数、排名为何变化，以及这种检索几何与知识库水印的关系。

## 知识点速查

- [1. 透明 BM25 与 Dense 对照实验](#1-透明-bm25-与-dense-对照实验)
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

## 小结

本实验完成了一个不依赖外部检索库的透明 BM25，并在与 Dense 完全相同的 12 个 Chunk 和 5 个问题上完成对照。BM25 的 Gold Answer Chunk Recall@1 为 1.0，修复了 Dense 在 q01 上“主题正确但证据不完整”的排名错误；与此同时，两种检索器的 Top-1 Chunk 一致率只有 0.4，证明它们使用的相关性信号确实不同。

最重要的结论不是当前小数据上谁的指标更高，而是已经建立了可解释对照：Dense 通过连续语义空间排序，BM25 通过 TF、IDF 和长度归一化排序。这个差异将直接决定水印短语的跨检索器迁移能力，也是后续 RRF Hybrid 和 Reranker 实验的基线。

## 参考资料

- [透明 Dense RAG：从文档切分到证据约束生成](./03-transparent-dense-rag.md)
- [BM25 检索器实现](../scripts/bm25_retriever.py)
- [BM25/Dense 对照实验入口](../scripts/run_bm25_retrieval.py)
- Robertson, S. E. and Zaragoza, H. *The Probabilistic Relevance Framework: BM25 and Beyond*. 2009.
