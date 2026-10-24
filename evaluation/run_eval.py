#!/usr/bin/env python3
"""在隔离存储里走真实入库链路建库，并对各检索配置做文档级评测。

``--storage`` 指向独立目录，不会触碰项目的 storage/。需要环境变量
SILICONFLOW_API_KEY（嵌入和重排序）。用法见 evaluation/README.md。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
KB_NAME = "eval_crud_rag"
KS = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="prepare_crud_rag.py 的输出目录")
    parser.add_argument("--storage", type=Path, required=True, help="评测专用存储目录")
    parser.add_argument("--output", type=Path, required=True, help="报告 JSON 路径")
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 题（0 为全部）")
    parser.add_argument("--index-timeout", type=int, default=3600)
    return parser.parse_args()


def configure_environment(storage: Path) -> None:
    """必须在导入应用模块之前调用，让所有路径指向评测专用存储。"""
    if not os.getenv("SILICONFLOW_API_KEY"):
        raise SystemExit("缺少环境变量 SILICONFLOW_API_KEY")
    storage.mkdir(parents=True, exist_ok=True)
    os.environ["STORAGE_DIR"] = str(storage)
    for name in ("DATA_DIR", "DB_FILE", "VECTOR_STORE_DIR", "CHROMA_PATH", "MINERU_RESULTS_DIR"):
        os.environ.pop(name, None)
    sys.path.insert(0, str(PROJECT_DIR))


def build_knowledge_base(corpus_dir: Path, timeout: int) -> dict:
    """创建知识库并上传语料；若已全部就绪则直接复用。返回建库统计。"""
    import app_storage
    from services.core.container import get_application_container
    from services.documents import use_cases as document_use_cases
    from services.indexing import worker as index_worker
    from services.knowledge_bases import use_cases as kb_use_cases

    app_storage.initialize_app_storage()
    container = get_application_container()
    files = sorted(corpus_dir.glob("*.txt"))
    wanted = {path.name for path in files}

    if not container.documents.knowledge_base_exists(KB_NAME):
        kb_use_cases.create_knowledge_base(KB_NAME)
    existing = {record["filename"] for record in container.documents.list_documents(KB_NAME)}
    started = time.perf_counter()
    index_worker.start()
    try:
        for number, path in enumerate(files, start=1):
            if path.name in existing:
                continue
            with path.open("rb") as handle:
                document_use_cases.upload_document(
                    KB_NAME, path.name, handle, metadata=None, parser_mode=None
                )
            if number % 100 == 0:
                print(f"  已提交上传：{number} / {len(files)}", flush=True)
        deadline = time.monotonic() + timeout
        while True:
            records = container.documents.list_documents(KB_NAME)
            pending = [r for r in records if r["status"] == "processing"]
            if not pending:
                break
            if time.monotonic() > deadline:
                raise SystemExit(f"索引超时，仍有 {len(pending)} 个文档在处理")
            print(f"  索引中：剩余 {len(pending)} / {len(records)}", flush=True)
            time.sleep(10)
    finally:
        index_worker.stop()

    records = container.documents.list_documents(KB_NAME)
    ready = {r["filename"] for r in records if r["status"] == "ready"}
    failed = sorted(wanted - ready)
    return {
        "documents": len(wanted),
        "ready": len(ready & wanted),
        "failed": failed,
        "index_seconds": round(time.perf_counter() - started, 1),
    }


def load_questions(data_dir: Path, limit: int) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (data_dir / "questions.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit > 0 else rows


def sources_of(documents: list) -> list[str]:
    return [str((doc.metadata or {}).get("source", "")) for doc in documents]


def interleave(groups: list[list]) -> list:
    """多条路线的结果轮流取一个，近似“各路线同等对待”的单后端排序。"""
    merged: list = []
    for index in range(max((len(g) for g in groups), default=0)):
        merged.extend(g[index] for g in groups if index < len(g))
    return merged


async def evaluate_question(row: dict, container) -> dict:
    """对一道题依次跑各检索配置，返回每个配置的来源排名和耗时（毫秒）。"""
    from services.chat import query as chat_query
    from services.chat import ranking as chat_ranking
    from services.retrieval import context as context_builder
    from services.retrieval import retriever as retrieval

    question = row["question"]
    outputs: dict[str, dict] = {}

    def record(name: str, ranked: list[str], started: float) -> None:
        outputs[name] = {
            "ranked": ranked,
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }

    with retrieval.request_source_snapshot(KB_NAME, container):
        capabilities = retrieval._probe_base_capabilities(KB_NAME, None)
        embeddings = retrieval.CachedQueryEmbeddings(container.create_query_embeddings())
        pairs = [
            retrieval._get_base_retrievers(KB_NAME, embeddings, capabilities, route)
            for route in sorted(capabilities.chroma_routes | capabilities.fts_routes)
        ]
        started = time.perf_counter()
        fts_docs = interleave(
            [await asyncio.to_thread(fts.invoke, question) for _, fts in pairs if fts]
        )
        record("fts_only", sources_of(fts_docs), started)

        started = time.perf_counter()
        vector_docs = interleave(
            [await asyncio.to_thread(vec.invoke, question) for vec, _ in pairs if vec]
        )
        record("vector_only", sources_of(vector_docs), started)

        started = time.perf_counter()
        ensemble = retrieval.build_retriever(KB_NAME, container=container)
        hybrid_docs = await asyncio.to_thread(ensemble.invoke, question) if ensemble else []
        record("hybrid", sources_of(hybrid_docs), started)

    started = time.perf_counter()
    candidates = await chat_query.retrieve_rerank_candidates(
        KB_NAME, question, False, None, container=container
    )
    record("hybrid_plus_lexical", sources_of(candidates), started)

    ranked = await chat_ranking.rerank(question, candidates, container=container)
    # 耗时从候选召回开始累计，反映加上重排序后的总成本。
    outputs["hybrid_rerank"] = {
        "ranked": sources_of(ranked),
        "ms": round((time.perf_counter() - started) * 1000, 1),
    }

    spans = await asyncio.to_thread(context_builder.build_context_spans, KB_NAME, ranked)
    outputs["final_context"] = {
        "ranked": [span.source for span in spans],
        "ms": round((time.perf_counter() - started) * 1000, 1),
    }
    return outputs


CONFIG_NOTES = {
    "fts_only": "仅 FTS5 关键词检索（各路线轮流合并）",
    "vector_only": "仅 Chroma 向量检索（各路线轮流合并）",
    "hybrid": "FTS5 + 向量加权融合（EnsembleRetriever）",
    "hybrid_plus_lexical": "融合 + 精确词补充 + 跨路线去重，取重排候选（上限 30）",
    "hybrid_rerank": "候选经 BAAI/bge-reranker-v2-m3 重排，取前 10",
    "final_context": "重排后恢复上下文，实际交给回答模型的片段（最多 8 段）",
}


async def run(args: argparse.Namespace) -> int:
    configure_environment(args.storage)
    from evaluation import metrics
    from services.chat import ranking as chat_ranking
    from services.core.container import get_application_container

    manifest = json.loads((args.data / "manifest.json").read_text("utf-8"))
    print("建库（复用已有索引则很快）...", flush=True)
    index_stats = build_knowledge_base(args.data / "corpus", args.index_timeout)
    if index_stats["failed"]:
        print(f"警告：{len(index_stats['failed'])} 个文档未就绪，指标会偏低", flush=True)

    questions = load_questions(args.data, args.limit)
    container = get_application_container()
    chat_ranking.start_rerank_client()
    per_config: dict[str, list[dict]] = {}
    latency: dict[str, list[float]] = {}
    # 检索器内部会吞掉调用异常并返回空列表；单独统计空结果，避免把故障误读为效果差。
    empty: dict[str, int] = {}
    errors: list[dict] = []
    try:
        for number, row in enumerate(questions, start=1):
            try:
                outputs = await evaluate_question(row, container)
            except Exception as exc:  # 单题失败不应中断整轮评测
                errors.append({"id": row["id"], "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{number}/{len(questions)}] {row['id']} 失败：{exc}", flush=True)
                continue
            for name, output in outputs.items():
                per_config.setdefault(name, []).append(
                    metrics.question_metrics(output["ranked"], row["gold"], KS)
                )
                latency.setdefault(name, []).append(output["ms"])
                if not output["ranked"]:
                    empty[name] = empty.get(name, 0) + 1
            if number % 20 == 0 or number == len(questions):
                print(f"[{number}/{len(questions)}] 已完成", flush=True)
    finally:
        await chat_ranking.close_rerank_client()

    from services.core import config as application_config

    summary = {
        name: {
            **{k: round(v, 4) for k, v in metrics.summarize(rows).items()},
            "latency_ms_p50": round(metrics.percentile(latency[name], 0.5), 1),
            "latency_ms_p95": round(metrics.percentile(latency[name], 0.95), 1),
            "questions": len(rows),
            "empty_results": empty.get(name, 0),
        }
        for name, rows in per_config.items()
    }
    report = {
        "metadata": {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset": manifest,
            "index": index_stats,
            "embedding_model": application_config.DEFAULT_EMBEDDING_MODEL,
            "rerank_model": application_config.DEFAULT_RERANK_MODEL,
            "rerank_top_n": chat_ranking.RERANK_TOP_N,
            "rerank_max_candidates": application_config.RETRIEVAL_RERANK_MAX_CANDIDATES,
            "chunk_size": application_config.DEFAULT_CHUNK_SIZE,
            "chunk_overlap": application_config.DEFAULT_CHUNK_OVERLAP,
            "ks": list(KS),
            "evaluated": len(questions) - len(errors),
            "errors": errors,
        },
        "configs": CONFIG_NOTES,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n配置                  hit@1  hit@5  rec@5  rec@10 full@10 MRR    P50ms   空结果题数")
    for name, values in summary.items():
        print(
            f"{name:<22}{values['hit@1']:<7.3f}{values['hit@5']:<7.3f}"
            f"{values['recall@5']:<7.3f}{values['recall@10']:<7.3f}"
            f"{values['full@10']:<8.3f}{values['mrr']:<7.3f}{values['latency_ms_p50']:<8.0f}{values['empty_results']}"
        )
    print(f"\n报告：{args.output}")
    return 1 if errors else 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
