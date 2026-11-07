# 检测与降低大语言模型的幻觉 (Detecting and Reducing Hallucinations in LLMs)

DTSC 710 课程项目。本仓库用**同一个模型、同一批问题**，对比四种"逐步加码"的策略，看哪一种能在不牺牲正确率的前提下，减少模型"一本正经地胡说"（幻觉）。

核心脚本：[run_experiment.py](run_experiment.py)

---

## 这个实验到底在干什么

用一个小型本地模型（默认 `qwen2.5:3b`，通过 Ollama 跑）回答一批事实性问答题，然后逐层给模型加"防胡说"的机制，测量每一层带来的变化。

四种条件（A→B→C→D，一层比一层多一个机制）：

| 条件 | 做法 | 想验证什么 |
|---|---|---|
| **A 直接回答** | 直接让模型答，不给任何提示 | 基线：模型自己能答对多少、胡说多少 |
| **B + 允许弃答** | 在 A 基础上加一句"不确定就回答 `I don't know`" | 让模型学会"示弱"能不能减少错误答案 |
| **C + 检索增强** | 在 B 基础上，检索相关 Wikipedia 段落塞进提示里 | 给它真实证据，能不能少编 |
| **D + 二次核查** | 再调一次模型，让它拿 C 的草稿答案对照段落，判定 **KEEP / CORRECT / WITHDRAW** | 让模型"自我复核"能不能进一步纠错 |

D 的三种判定：
- **KEEP**：段落支持草稿答案 → 保留
- **CORRECT**：段落显示另一个答案 → 用新答案替换
- **WITHDRAW**：段落与草稿矛盾、或无法判断 → 改为弃答（`I don't know`）

---

## 数据与流程

1. **问题集**：Natural Questions 开放域开发集（`NQ-open.dev.jsonl`，首次运行自动下载到 `data/`）。会过滤掉：
   - 时效性问题（含 current / latest / now 等词）——避免答案随时间变化
   - 答案过长（> 5 词）的题
   - yes/no 类问题

   然后用固定随机种子（`--seed`，默认 42）打乱，取前 `--n` 题。

2. **语料库（闭集检索）**：对每道题，用**问题文本**（绝不用标准答案）去搜 Wikipedia，抓回若干页面，正文切成 100 词的小段落，汇成一个封闭语料库（`results/corpus.jsonl`）。
   > 关键设计：检索只用问题、不用答案，避免"作弊"泄露标准答案。

3. **检索器**：对语料库建 **BM25** 索引（`rank-bm25`），每题取 top-`k`（默认 `k=5`）段落作为证据。

4. **逐题运行**：每道题依次跑 A / B / C / D 四种条件，完整记录原始输出、清洗后的答案、token 数、调用次数等，逐行写入 `results/records.jsonl`（**支持断点续跑**——中断后重跑会跳过已完成的题）。

5. **判分与汇总**：
   - 答案归一化（小写、去标点、去冠词、数字词转阿拉伯数字、处理 US/UK/NYC 等别名）后与标准答案比对
   - 每个答案打标签：`correct` / `incorrect` / `abstain`（弃答）
   - 计算各条件的正确率、错误率、弃答率、**选择性准确率**（答了的题里答对的比例）、平均调用次数与 token 开销、检索 Recall@k
   - 用 **bootstrap** 给相邻条件的差异算 95% 置信区间

---

## 输出文件（都在 `results/`）

| 文件 | 内容 |
|---|---|
| `records.jsonl` | 每题每条件的完整明细（原始输出、答案、标签、token、检索到的段落 id 等） |
| `summary.json` | 全部统计指标的机器可读版 |
| `tables.md` | 人读版结果表：四条件对比表、核查器决策表、相邻条件差异的置信区间，以及几个诊断数字 |
| `corpus.jsonl` / `corpus_meta.json` | 构建好的闭集语料库及其元信息 |
| `annotation_sample.csv` | 随机抽 100 题的样本，供两位作者人工标注核对（含 annotator1/2 列） |

---

## 如何运行（macOS）

```bash
# 1) 安装并启动 Ollama，拉取模型
#    https://ollama.com
ollama pull qwen2.5:3b

# 2) 装 Python 依赖（建议用虚拟环境）
python3 -m venv .venv
.venv/bin/pip install requests rank-bm25

# 3) 跑实验
.venv/bin/python -u run_experiment.py --n 100
#    结果输出到 ./results/
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--n` | 300 | 抽取的题目数 |
| `--k` | 5 | 检索 top-k 段落数 |
| `--seed` | 42 | 抽样随机种子 |
| `--model` | `qwen2.5:3b` | Ollama 模型名 |
| `--host` | `http://localhost:11434` | Ollama 服务地址 |
| `--pages-per-q` | 3 | 每题从 Wikipedia 抓的页面数 |
| `--out` | `results` | 输出目录 |
| `--dry-run` | - | 离线冒烟测试（用假语料 + 随机假模型，仅验证流程，**结果无意义**） |

> 为了可复现，模型调用固定 `temperature=0, seed=0`。

---

## 关于 Wikipedia 抓取的速率限制

构建语料库时需要大量请求 Wikipedia API。请求过快会触发 **HTTP 429 (Too Many Requests)**，导致页面几乎抓不到、语料库为空。

脚本内置了礼貌抓取机制（`wiki_get`）：
- 请求之间最小间隔 **1.2 秒**（约 0.8 请求/秒，对 search 接口是安全速率）
- 带 `maxlag=5` 参数，让服务器在副本延迟时优雅限流
- 遇到 429 / 503 时按 `Retry-After`（或指数退避，上限 30s）重试，最多 6 次

代价是语料库构建偏慢（约 20–40 分钟，取决于 `--n`），但能保证抓取稳定、语料库完整。

---

## 依赖

- Python 3.9+
- [`requests`](https://pypi.org/project/requests/)、[`rank-bm25`](https://pypi.org/project/rank-bm25/)
- [Ollama](https://ollama.com) 及一个本地模型（默认 `qwen2.5:3b`）
