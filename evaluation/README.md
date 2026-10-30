# 离线检索评测

用公开数据集 [CRUD-RAG](https://github.com/IAAR-Shanghai/CRUD_RAG)（Apache-2.0）评测检索质量。语料经过真实的上传、解析、分块、写入 Milvus 的入库链路，再对不同检索配置做文档级对比。

## 范围与局限

- 只评测文本新闻检索。表格、版面、OCR 和 MinerU 路线没有被量化。
- 语料是抽样的：标准文档加随机干扰文档，规模写在 `manifest.json`。语料比真实场景小，绝对数值会偏乐观，更适合看配置之间的相对差异。
- 题目取自 CRUD-RAG 的 2 文档和 3 文档问答任务，标准答案是数据集自带的原文，不是人工标注的本项目数据。
- 指标按“文档”计算：多个切片命中同一文档只算一次。

## 运行

1. 获取数据集并抽样（只用标准库，输出目录必须为空）：

```bash
git clone --depth 1 https://github.com/IAAR-Shanghai/CRUD_RAG.git /tmp/crud_rag_src/repo
python3 -I evaluation/prepare_crud_rag.py \
  --repo /tmp/crud_rag_src/repo --output /tmp/crud_eval_data/set \
  --questions 200 --distractors 800 --seed 42
```

如需让固定和结构化两条路线同时入库，给第 1 步加 `--extension md`（默认 `txt` 只走固定路线）。

2. 建库并评测。使用独立存储目录，不会触碰项目的 `storage/`；嵌入和重排序会调用 SiliconFlow，需要 `SILICONFLOW_API_KEY`。包装脚本会把 `/tmp/crud_eval_data`（可用 `EVAL_DIR` 改）挂进后端容器，额外参数原样传给 `run_eval.py`：

```bash
export SILICONFLOW_API_KEY=...
./evaluation/run_docker.sh            # 全量
./evaluation/run_docker.sh --limit 20 # 小规模试跑
```

等价的手动命令：

```bash
docker compose run --rm --no-deps -v "$PWD:/app" \
  -v /tmp/crud_eval_data:/eval -e SILICONFLOW_API_KEY \
  -e PYTHONPYCACHEPREFIX=/tmp/pymodel-pycache backend \
  python evaluation/run_eval.py \
  --data /eval/set --storage /eval/storage --output /eval/report.json
```

索引完成后再次运行会复用已建好的库。加 `--limit 20` 可先小规模试跑。

可选参数：`--rerank-top-n N` 只在本次评测里覆盖重排保留数；`--context-spans N`、`--context-chars N` 覆盖最终上下文的段数和字符上限；`--agentic` 额外评测补充检索（`CHAT_AGENTIC_RETRIEVAL`），会调用对话模型，默认用 `Qwen/Qwen2.5-7B-Instruct`，可用环境变量 `EVAL_AGENTIC_MODEL` 改。

## 对比的配置

| 配置 | 含义 |
| --- | --- |
| `fts_only` | 仅 Milvus BM25 关键词检索 |
| `vector_only` | 仅 Milvus 稠密向量检索 |
| `hybrid` | Milvus hybrid_search（稠密 + BM25），默认 RRF 融合，`--ranker weighted` 改为加权融合 |
| `hybrid_plus_lexical` | 融合、精确词补充和跨路线去重，取重排候选（上限 30） |
| `hybrid_rerank` | 候选经重排序后取前 `RERANK_TOP_N`（默认 20，可用 `--rerank-top-n` 覆盖） |
| `final_context` | 恢复上下文后实际交给回答模型的片段（最多 8 段） |

## 指标

- `hit@k`：前 k 个文档中至少含一篇标准文档的题目比例。
- `recall@k`：前 k 个文档覆盖的标准文档比例，再对题目取平均。
- `full@k`：前 k 个文档覆盖全部标准文档的题目比例（多文档题的关键指标）。
- `mrr`：首个标准文档排名的倒数的平均值。
- `latency_ms_p50` / `p95`：该配置单题耗时。`hybrid_rerank` 和 `final_context` 的耗时从候选召回开始累计，包含远程重排序。

报告写入 `--output`，并记录数据集版本、抽样种子、模型名和分块参数，便于复现。

## 已提交的报告

`results/` 下的报告都是 200 题、1223 篇文档、种子 42 的同一份抽样：

| 文件 | 内容 |
| --- | --- |
| `crud_rag_report_top20_agentic.json` | 当前线上配置（`RERANK_TOP_N = 20`），另含补充检索对照（`--agentic`）和重排覆盖诊断，根目录 README 的表格取自这份 |
| `crud_rag_report.json` | 切换到 Milvus 后的首轮结果，当时 `RERANK_TOP_N = 10` |
| `crud_rag_report_chroma_fts5_baseline.json` | 切换前 Chroma + SQLite FTS5 的基线 |

结论与解读：

- 重排保留数从 10 提到 20 的依据：top_n=10 时重排后 full@10 从 0.845 降到 0.775，丢失的 14 题全部是重排只返回 10 个切片、去重后不足 10 篇文档，属于截断造成的覆盖损失；top_n=20 时丢失数为 0。
- 补充检索在这份数据上 hit@1、MRR 与关闭时相同，最终上下文 full@10 从 0.835 降到 0.820，P50 延迟从 414 ms 升到 1354 ms，因此默认关闭。
- Milvus BM25 的 MRR 高于旧 FTS5（0.896 对 0.826）。两次运行之间还改过其他环节，差异不能全部归因于检索库。
- 200 题的抽样误差约 4 到 6 个百分点（二项分布 95% 区间粗估，没有重复运行），混合与仅向量、重排与不重排之间的 MRR 差异都在这个范围内。
