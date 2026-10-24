# 离线检索评测

用公开数据集 [CRUD-RAG](https://github.com/IAAR-Shanghai/CRUD_RAG)（Apache-2.0）评测检索质量。语料经过真实的上传、解析、分块、写入 Chroma 和 FTS5 的入库链路，再对不同检索配置做文档级对比。

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

## 对比的配置

| 配置 | 含义 |
| --- | --- |
| `fts_only` | 仅 FTS5 关键词检索 |
| `vector_only` | 仅 Chroma 向量检索 |
| `hybrid` | FTS5 与向量加权融合 |
| `hybrid_plus_lexical` | 融合、精确词补充和跨路线去重，取重排候选（上限 30） |
| `hybrid_rerank` | 候选经重排序后取前 10 |
| `final_context` | 恢复上下文后实际交给回答模型的片段（最多 8 段） |

## 指标

- `hit@k`：前 k 个文档中至少含一篇标准文档的题目比例。
- `recall@k`：前 k 个文档覆盖的标准文档比例，再对题目取平均。
- `full@k`：前 k 个文档覆盖全部标准文档的题目比例（多文档题的关键指标）。
- `mrr`：首个标准文档排名的倒数的平均值。
- `latency_ms_p50` / `p95`：该配置单题耗时。`hybrid_rerank` 和 `final_context` 的耗时从候选召回开始累计，包含远程重排序。

报告写入 `--output`，并记录数据集版本、抽样种子、模型名和分块参数，便于复现。
