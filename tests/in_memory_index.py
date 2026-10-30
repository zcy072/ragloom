"""ChunkIndex 的内存实现，仅用于单元测试：稠密用余弦，全文用简化 BM25。"""

import math
import re
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Sequence

from langchain_core.documents import Document as LCDocument

from services.indexing.chunking import (
    assign_chunk_indexes,
    chunk_route,
    contextualized_chunk_text,
)
from services.retrieval.chunk_index import record_id

_WORD = re.compile(r"[A-Za-z0-9]+")
_CJK = re.compile(r"[一-鿿]")
_RRF_K = 60


def _tokens(text: str) -> list[str]:
    """英文数字按词切，中文按单字切，足够区分测试用例。"""
    return [word.lower() for word in _WORD.findall(text)] + _CJK.findall(text)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


class InMemoryChunkIndex:
    """线程安全的内存索引，行为对齐 MilvusChunkIndex 的公开语义。"""

    def __init__(self) -> None:
        """创建空索引。"""
        self._lock = threading.RLock()
        self._rows: list[dict] = []
        self.fail_next_publish: Exception | None = None

    def _match(self, row, kb_id, route, sources) -> bool:
        """判断一行是否满足知识库、路线和文件范围。"""
        if row["kb_id"] != kb_id:
            return False
        if route and row["chunk_route"] != route:
            return False
        return sources is None or row["source"] in set(sources)

    @staticmethod
    def _document(row: dict) -> LCDocument:
        """把内部行转换为带检索元数据的 Document。"""
        metadata = {
            key: row[key]
            for key in (
                "source", "page", "chunk_index", "chunk_route", "parent_id",
                "contextual_title", "visual_kind", "visual_image", "visual_analysis",
            )
            if row.get(key) not in (None, "")
        }
        return LCDocument(page_content=row["content"], metadata=metadata)

    def publish_source(self, kb_id, source, splits, vectors) -> None:
        """先追加新代，再删除旧代；注入的失败会回滚新代并抛出。"""
        splits = assign_chunk_indexes(splits)
        if len(vectors) != len(splits):
            raise ValueError(f"向量数量与分块数量不一致：{len(vectors)} != {len(splits)}")
        generation = uuid.uuid4().hex
        new_rows = []
        for position, (split, vector) in enumerate(zip(splits, vectors)):
            meta = split.metadata
            new_rows.append(
                {
                    "id": record_id(kb_id, source, position, split, generation),
                    "kb_id": kb_id,
                    "source": source,
                    "generation": generation,
                    "dense": list(vector),
                    "search_text": contextualized_chunk_text(split),
                    "content": split.page_content,
                    "page": str(meta.get("page", meta.get("row", ""))),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                    "chunk_route": chunk_route(split),
                    "parent_id": meta.get("parent_id", ""),
                    "contextual_title": meta.get("contextual_title", ""),
                    "visual_kind": meta.get("visual_kind", ""),
                    "visual_image": meta.get("visual_image", ""),
                    "visual_analysis": meta.get("visual_analysis", ""),
                }
            )
        with self._lock:
            if self.fail_next_publish is not None:
                error, self.fail_next_publish = self.fail_next_publish, None
                raise error
            self._rows.extend(new_rows)
            self._rows = [
                row
                for row in self._rows
                if not (
                    row["kb_id"] == kb_id
                    and row["source"] == source
                    and row["generation"] != generation
                )
            ]

    def delete_source(self, kb_id, source) -> None:
        """删除一个文件的全部子块。"""
        with self._lock:
            self._rows = [
                row for row in self._rows
                if not (row["kb_id"] == kb_id and row["source"] == source)
            ]

    def delete_kb(self, kb_id) -> None:
        """删除一个知识库的全部子块。"""
        with self._lock:
            self._rows = [row for row in self._rows if row["kb_id"] != kb_id]

    def delete_stale_sources(self, kb_id, active_sources) -> int:
        """删除不在活跃集合里的文件子块。"""
        with self._lock:
            before = len(self._rows)
            self._rows = [
                row for row in self._rows
                if row["kb_id"] != kb_id or row["source"] in active_sources
            ]
            return before - len(self._rows)

    def ensure_collection(self) -> None:
        """内存实现无需准备。"""

    def check_ready(self, timeout: float) -> str:
        """内存实现始终就绪。"""
        return "in-memory"

    def reset_if_empty(self, has_documents: Callable[[], bool]) -> bool:
        """没有任何文档时清空全部行。"""
        if has_documents():
            return False
        with self._lock:
            if has_documents():  # 与真实索引一致：拿到锁后再确认一次
                return False
            self._rows = []
        return True

    def _dense_ranked(self, rows, query_vector) -> list[dict]:
        """按余弦相似度降序排列。"""
        return sorted(rows, key=lambda row: -_cosine(query_vector, row["dense"]))

    def _bm25_ranked(self, rows, query: str) -> list[dict]:
        """用候选集内的简化 BM25 排序，只返回得分大于零的行。"""
        query_terms = set(_tokens(query))
        if not rows or not query_terms:
            return []
        docs = [Counter(_tokens(row["search_text"])) for row in rows]
        average = sum(sum(doc.values()) for doc in docs) / len(docs) or 1.0
        scored = []
        for row, doc in zip(rows, docs):
            length = sum(doc.values())
            score = 0.0
            for term in query_terms:
                frequency = doc.get(term, 0)
                if not frequency:
                    continue
                containing = sum(1 for other in docs if term in other)
                idf = math.log(1 + (len(docs) - containing + 0.5) / (containing + 0.5))
                score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average))
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: -item[0])
        return [row for _, row in scored]

    def _candidates(self, kb_id, route, sources) -> list[dict]:
        """在锁内复制满足过滤条件的行。"""
        with self._lock:
            return [row for row in self._rows if self._match(row, kb_id, route, sources)]

    def search_dense(self, kb_id, query_vector, route, sources, k) -> list[LCDocument]:
        """只做稠密检索。"""
        rows = self._dense_ranked(self._candidates(kb_id, route, sources), query_vector)
        return [self._document(row) for row in rows[:k]]

    def search_lexical(self, kb_id, query, route, sources, k) -> list[LCDocument]:
        """只做 BM25 检索。"""
        rows = self._bm25_ranked(self._candidates(kb_id, route, sources), query)
        return [self._document(row) for row in rows[:k]]

    def search_hybrid(self, kb_id, query, query_vector, route, sources, k) -> list[LCDocument]:
        """稠密与 BM25 两路结果做 RRF 融合。"""
        rows = self._candidates(kb_id, route, sources)
        scores: dict[str, float] = {}
        by_id = {row["id"]: row for row in rows}
        for ranked in (
            self._dense_ranked(rows, query_vector)[:k],
            self._bm25_ranked(rows, query)[:k],
        ):
            for rank, row in enumerate(ranked):
                scores[row["id"]] = scores.get(row["id"], 0.0) + 1 / (_RRF_K + rank + 1)
        ordered = sorted(scores, key=lambda row_id: -scores[row_id])[:k]
        return [self._document(by_id[row_id]) for row_id in ordered]

    def available_routes(self, kb_id, sources) -> set[str]:
        """返回这些文件下存在的路线。"""
        return {row["chunk_route"] for row in self._candidates(kb_id, None, sources)}

    def neighbors(self, kb_id, source, route, lo, hi) -> list[LCDocument]:
        """按 chunk_index 闭区间取相邻子块。"""
        rows = [
            row for row in self._candidates(kb_id, route, [source])
            if lo <= row["chunk_index"] <= hi
        ]
        rows.sort(key=lambda row: row["chunk_index"])
        return [self._document(row) for row in rows]
