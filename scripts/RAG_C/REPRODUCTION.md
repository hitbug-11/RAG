# RAG© 补充代码复现说明

本文档记录对论文补充材料的代码审计，以及当前已经验证的复现入口。原始附件保留在本目录；新增脚本不依赖原始 `run.py`。

## 当前复现范围

已完成 NQ 全部 100 个验证问题的 Contriever 检索门控复现：

```text
普通问题 ──────────→ non-target CoT
水印问题 + 水印短语 → target CoT
```

这一步验证水印短语能否把 target CoT 拉入 Top-k。它是生成 VSR 和 Wilcoxon 所有权检验的上游条件，不能直接替代这两个论文指标。

## 为什么不直接运行原始入口

补充材料更接近匿名审稿代码快照，而不是完整发布仓库：

- `README.md` 只列出宽松依赖，未固定 BEIR、Transformers、Contriever 和生成模型 revision；
- 原始环境要求 PyTorch 1.13 + CUDA 11.7，当前服务器环境为 PyTorch 2.6 + CUDA 12.4；
- `run.py`、`main.py` 和 `parse.py` 混有 NQ/HotpotQA 文件名硬编码；
- `main.py` 依赖占位 API key 和较旧的模型 ID，且没有完整保存论文指标；
- RAG©-O 所需的 HotFlip/联合优化代码在 `src/attack.py` 中被注释或留空；
- 发布的部分结果 JSON 已将生成文本替换为判定标签，无法据此重新计算 Answer Accuracy；
- 原代码把干净语料的预计算分数与注入 CoT 分数合并，但没有提供统一、可审计的排名 Trace。

因此新增 `reproduce_retrieval.py`，只复现附件当前足以独立验证的检索门控，并保存逐问题分数和排名。

## 输入与模型

- Dataset：附件提供的 100 个 NQ 验证问题；
- Clean competitors：附件提供的 Contriever Top-100 分数；
- Injected texts：每个问题对应的 target CoT 与 non-target CoT；
- Retriever：`facebook/contriever`；
- 固定 revision：`2bd46a25019aeea091fd42d1f0fd4801675cf699`；
- Score：未归一化 Embedding 的 dot product，与论文和原代码一致；
- Top-k：1、3、5、10；
- GPU：单张 NVIDIA L20。

脚本同时保存四份输入文件的 SHA-256、Python/PyTorch/Transformers/CUDA 版本和 GPU 名称。

## 运行方法

本地只做输入和排名逻辑检查：

```bash
python3 -m unittest discover -s scripts/RAG_C/tests -v
python3 scripts/RAG_C/reproduce_retrieval.py \
  --validate-only \
  --num-questions 100 \
  --output /tmp/ragc_validate.json
```

服务器运行：

```bash
HF_HUB_OFFLINE=1 bash scripts/run_server_python.sh \
  scripts/RAG_C/reproduce_retrieval.py \
  --num-questions 100 \
  --seed 12 \
  --device cuda:0 \
  --output results/ragc_reproduction/nq_contriever_retrieval_gate_100.json
```

## 实测结果

| k | 普通问题 target 泄漏率 | 普通问题 non-target Hit | 水印问题 target Hit | 严格门控成功率 |
|---:|---:|---:|---:|---:|
| 1 | 0.07 | 0.41 | 0.94 | 0.38 |
| 3 | 0.28 | 0.55 | 0.97 | 0.33 |
| 5 | 0.37 | 0.61 | 0.98 | 0.32 |
| 10 | 0.47 | 0.69 | 0.98 | 0.31 |

主要结论：

1. 水印问题的 target CoT Hit@5 达到 0.98，核心检索增强现象复现成功；
2. `test208` 和 `test107` 的 target CoT 在水印问题下分别排第 50 和第 12，是 Top-5 的两个失败案例；
3. 普通问题的 target CoT Top-5 泄漏率为 0.37，说明 target CoT 仍与原问题共享较强答案语义；
4. 若同时要求“水印问题命中 target、普通问题命中 non-target、普通问题不命中 target”，Top-5 严格门控成功率只有 0.32；
5. 论文报告的 NQ RAG©-L GPT-4 VSR 为 0.86，而这里的 0.98 是 Retriever Hit@5，两者定义不同，不能据数值高低判断是否复现论文 Table 1。

完整逐问题结果见 [`nq_contriever_retrieval_gate_100.json`](../../results/ragc_reproduction/nq_contriever_retrieval_gate_100.json)。

## 尚未完成

- 使用固定 Generator 重新生成普通/水印问题的输出；
- 比较 target CoT、non-target CoT 和最终输出之间的语义信息；
- Answer Accuracy 与 Harmfulness；
- Judge Prompt 稳定性和人工抽样复核；
- 配对 Wilcoxon 检验、FPR、效应量与置信区间；
- RAG©-O 优化代码的重建与验证。

在这些步骤完成前，本结果应称为“检索门控复现”，不能称为论文端到端结果复现。

## 论文端到端路线

用户已指定继续采用论文路线，不以 Qwen 或其他本地模型替代。新增
`reproduce_end_to_end.py`，固定以下默认配置：

- Contriever Top-5；
- Generator：`gpt-4-0613`；
- Generator Prompt：附件 `src/prompts.py` 中的原始模板；
- Generator 温度：0.1，seed：100；
- Detector：`gpt-4-0613`；
- Detector Prompt：论文附录 B.3 的 Yes/No 模板；
- 指标：VSR、普通问题 target FPR、Answer Accuracy、Harmfulness；
- 所有权验证：单侧配对 Wilcoxon，显著性水平 0.01；
- VSR、FPR 与 Answer Accuracy 同时保存 95% Wilson 区间；
- 额外保存 10/20/50/100 问前缀检验结果。

四个阶段相互独立，生成与 Judge 使用追加式 JSONL 检查点：

```bash
python scripts/RAG_C/reproduce_end_to_end.py prepare \
  --corpus /data/haojiachen/rag/data/beir/nq.jsonl.gz \
  --output results/ragc_reproduction/nq_paper_generation_inputs.json

OPENAI_API_KEY=... python scripts/RAG_C/reproduce_end_to_end.py generate \
  --input results/ragc_reproduction/nq_paper_generation_inputs.json \
  --output results/ragc_reproduction/nq_gpt4_0613_generations.jsonl

OPENAI_API_KEY=... python scripts/RAG_C/reproduce_end_to_end.py judge \
  --input results/ragc_reproduction/nq_paper_generation_inputs.json \
  --generations results/ragc_reproduction/nq_gpt4_0613_generations.jsonl \
  --output results/ragc_reproduction/nq_gpt4_0613_judgments.jsonl

python scripts/RAG_C/reproduce_end_to_end.py evaluate \
  --input results/ragc_reproduction/nq_paper_generation_inputs.json \
  --generations results/ragc_reproduction/nq_gpt4_0613_generations.jsonl \
  --judgments results/ragc_reproduction/nq_gpt4_0613_judgments.jsonl \
  --output results/ragc_reproduction/nq_paper_end_to_end_metrics.json
```

API key 只从环境变量读取，不写入配置或结果。程序不会在
`gpt-4-0613` 不可用时自动换模型，避免把替代实验误报为论文复现。
恢复已有 JSONL 时会校验模型、温度和 seed，拒绝把不同配置的输出混入
同一实验。

### 输入准备实测

BEIR NQ 语料使用公开 `nq.jsonl.gz` 镜像，SHA-256 为
`83ee077d2065a4e95f892e1efcc5d0cbce9310651eecbd8bb3825dc3378ff377`。
100 个问题已形成 200 条普通/水印 Prompt：

| 条件 | clean 槽位 | target CoT 槽位 | non-target CoT 槽位 |
|---|---:|---:|---:|
| 普通问题 | 402 | 37 | 61 |
| 水印问题 | 349 | 98 | 53 |

全部 1000 个 Top-5 槽位中，clean、target CoT、non-target CoT 分别为
751、135、114。程序执行了 400 次“候选 Rank≤5 与上下文实际出现”
一致性检查，错误数为 0。完整 Prompt 与上下文来源保存在
`results/ragc_reproduction/nq_paper_generation_inputs.json`。

### Wilcoxon 公式审计

论文命题 3.3 印刷为：

```text
H0: C(X′) + C(X) = 0
H1: C(X′) + C(X) > 0
```

但论文期望的理想行为是 `C(X′)=1`、`C(X)=-1`，代入加号后恰好为 0，
无法拒绝原假设。这与“比较水印问题和普通问题”以及 Table 2 的极小
p-value 不一致。代码因此把可执行主检验实现为标准配对差
`C(X′)-C(X)>0`，并在结果中额外保存原文加号公式的逐对取值分布，
不静默掩盖论文正文中的符号矛盾。

### 当前阻塞

截至 2026-07-27，本地和服务器均未配置 `OPENAI_API_KEY`，因此尚未发送
真实 GPT 请求。`gpt-4-0613` 在当前 OpenAI 模型文档中已标记为
deprecated；是否仍可调用取决于具体项目权限。在获得具备访问权限的
环境变量前，只完成输入准备和 12 个离线流程测试，不能产生或声称论文
VSR。服务器也没有论文 LLaMA-3(8B) 权重或 Hugging Face token，因此
当前不存在无需变更模型即可启动的论文内替代 Generator。
