"""SQLite FTS5 索引维护和关键词检索。"""

import logging
import os
import re
import shutil
import sqlite3
import uuid
from collections.abc import Callable
from typing import List

import jieba
from jieba import analyse as jieba_analyse
from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from services.core import sqlite_connections as application_db
from services.core.settings import get_settings
from services.indexing import schema as index_schema
from services.indexing.chunking import (
    CHUNK_ROUTE_FIXED,
    CHUNK_ROUTE_STRUCTURED,
    CHUNK_ROUTE_VISUAL,
    assign_chunk_indexes,
    chunk_route,
    contextualized_chunk_text,
)
from services.retrieval import source_state

logger = logging.getLogger(__name__)

PARENT_CHUNK_TABLE = "parent_chunks"
_FTS_SEARCHABLE_TOKEN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
# 单字虚词几乎出现在每个片段里，放进 OR 条件只会扩大命中集合、增加排序噪声。
# 只过滤这一小批单字；多字词和数字、英文单字符（如型号里的 A）不受影响。
_FTS_SINGLE_CHAR_STOPWORDS = frozenset(
    "的是了和与及或在有也就都而且把被将对于为以其之呢吗吧啊么个这那"
)
_FTS_KEYWORD_POS = ("n", "nr", "ns", "nt", "nz", "vn", "v", "eng", "m", "a")
_FTS_SELECT_COLUMNS = (
    "content, source, page, original_text, chunk_index, parent_id, "
    "contextual_title, chunk_route"
)


def vector_store_dir(kb_name: str) -> str:
    """返回指定知识库在向量存储根目录下的索引目录。"""
    return os.path.join(get_settings().vector_store_dir, kb_name)


def fts_db_path(kb_name: str) -> str:
    """返回指定知识库的 FTS5 数据库路径。"""
    return os.path.join(vector_store_dir(kb_name), "fts5.db")


def _fts_phrase_term(value: str) -> str:
    """将一个词安全包装为 FTS5 精确短语项。"""
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _fts_match_queries(query: str, limit: int) -> list[tuple[str, int]]:
    """优先使用精确关键词，同时保留较宽松的词法召回作为补充。"""
    tokens: list[str] = []
    for token in jieba.cut(query):
        token = token.strip()
        if (
            token
            and token not in _FTS_SINGLE_CHAR_STOPWORDS
            and _FTS_SEARCHABLE_TOKEN.search(token)
            and token not in tokens
        ):
            tokens.append(token)
    exact_phrases = [
        phrase
        for phrase in re.findall(r"[a-zA-Z0-9\-]+", query)
        if len(phrase) >= 4 and phrase not in tokens
    ]
    broad_terms = [_fts_phrase_term(token) for token in [*tokens, *exact_phrases]]
    if not broad_terms:
        return []

    try:
        extracted = jieba_analyse.extract_tags(query, topK=4, allowPOS=_FTS_KEYWORD_POS)
    except Exception as exc:
        logger.debug("FTS keyword extraction fallback: %s", exc)
        extracted = []
    token_set = set(tokens)
    strict_tokens: list[str] = []
    for token in extracted:
        token = str(token).strip()
        if token in token_set and token not in strict_tokens:
            strict_tokens.append(token)

    broad_query = " OR ".join(broad_terms)
    if len(strict_tokens) < 2:
        return [(broad_query, limit)]
    strict_query = " AND ".join(_fts_phrase_term(token) for token in strict_tokens[:2])
    return [(strict_query, max(1, limit // 2)), (broad_query, limit)]


class FTS5Retriever(BaseRetriever):
    """使用 Jieba 中文分词的 SQLite FTS5 检索器。"""

    db_path: str = Field(...)
    k: int = Field(default=20)
    allowed_sources: tuple[str, ...] | None = Field(default=None)
    kb_name: str = Field(default="")
    chunk_route: str | None = Field(default=None)
    enforce_current_sources: bool = Field(default=False)

    def _get_relevant_documents(self, query: str) -> List[LCDocument]:
        """执行 FTS5 查询，并将命中行恢复为带检索元数据的文档。"""
        match_queries = _fts_match_queries(query, self.k)
        if not match_queries:
            return []

        try:
            allowed_sources = self.allowed_sources
            if self.enforce_current_sources:
                allowed_sources = source_state.current_source_allowlist(
                    self.kb_name, allowed_sources
                )
                if not allowed_sources:
                    return []
            with application_db.connect_fts5(self.db_path) as conn:
                if not fts5_index_is_current(self.db_path, conn=conn):
                    return []

                rows = _fetch_route_rows(
                    conn,
                    match_queries,
                    allowed_sources,
                    self.chunk_route,
                    self.k,
                )
        except Exception as exc:
            logger.warning("FTS5 search failed: %s", exc)
            return []

        return _rows_to_documents(rows, self.kb_name)

    async def _aget_relevant_documents(self, query: str) -> List[LCDocument]:
        """异步接口复用同步 FTS5 查询实现。"""
        return self._get_relevant_documents(query)


def _rows_to_documents(rows: list, kb_name: str) -> list[LCDocument]:
    """将 FTS5 查询结果行转换为 LangChain Document 列表。"""
    documents = []
    for row in rows:
        metadata: dict = {"source": row[1], "page": row[2]}
        if kb_name:
            metadata["kb_name"] = kb_name
        try:
            metadata["chunk_index"] = int(row[4])
        except (TypeError, ValueError):
            pass
        if row[5]:
            metadata["parent_id"] = row[5]
        if row[6]:
            metadata["contextual_title"] = row[6]
        if row[7]:
            metadata["chunk_route"] = row[7]
        documents.append(LCDocument(page_content=row[3], metadata=metadata))
    return documents


def _fetch_route_rows(
    conn: sqlite3.Connection,
    match_queries: list[tuple[str, int]],
    allowed_sources: tuple[str, ...] | list[str] | None,
    route: str | None,
    limit: int,
) -> list:
    """针对可选分块路线执行统一的 FTS 查询循环。"""
    rows: list = []
    seen_rows: set = set()
    for match_query, query_limit in match_queries:
        statement = f"SELECT {_FTS_SELECT_COLUMNS} FROM docs WHERE docs MATCH ?"
        params: list[str | int] = [match_query]
        if allowed_sources is not None:
            if not allowed_sources:
                return []
            placeholders = ", ".join("?" for _ in allowed_sources)
            statement += f" AND source IN ({placeholders})"
            params.extend(allowed_sources)
        if route:
            statement += " AND chunk_route = ?"
            params.append(route)
        statement += " ORDER BY bm25(docs) LIMIT ?"
        params.append(query_limit)
        for row in conn.execute(statement, params).fetchall():
            if row in seen_rows:
                continue
            rows.append(row)
            seen_rows.add(row)
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    return rows


def retrieve_all_routes(
    db_path: str,
    query: str,
    kb_name: str,
    allowed_sources: tuple[str, ...] | None,
    per_route: int,
    *,
    enforce_current_sources: bool = False,
) -> list[LCDocument]:
    """单次 SQLite 连接完成所有分块路线的关键词检索。

    相比为每条路线单独创建 FTS5Retriever，省去了重复打开连接和校验索引版本的开销；
    每条路线仍独立限制候选数量，保证来源多样性。
    """
    if not os.path.exists(db_path):
        return []
    match_queries = _fts_match_queries(query, per_route)
    if not match_queries:
        return []

    effective_sources = allowed_sources
    if enforce_current_sources:
        effective_sources = source_state.current_source_allowlist(
            kb_name, effective_sources
        )
    if effective_sources is not None and not effective_sources:
        return []

    all_documents: list[LCDocument] = []
    try:
        with application_db.connect_fts5(db_path) as conn:
            if not fts5_index_is_current(db_path, conn=conn):
                return []
            for route in (CHUNK_ROUTE_FIXED, CHUNK_ROUTE_STRUCTURED, CHUNK_ROUTE_VISUAL):
                rows = _fetch_route_rows(
                    conn,
                    match_queries,
                    effective_sources,
                    route,
                    per_route,
                )
                all_documents.extend(_rows_to_documents(rows, kb_name))
    except Exception as exc:
        logger.warning("FTS5 multi-route search failed: %s", exc)
        return []
    return all_documents



def fts5_index_version(
    db_path: str, *, conn: sqlite3.Connection | None = None
) -> int | None:
    """读取显式 FTS 架构版本，不会因检查而创建不存在的数据库。"""
    if not os.path.exists(db_path):
        return None
    try:
        if conn is not None:
            row = conn.execute("PRAGMA user_version").fetchone()
        else:
            with application_db.connect_fts5(db_path) as local_conn:
                row = local_conn.execute("PRAGMA user_version").fetchone()
    except sqlite3.Error:
        return None
    try:
        return int(row[0]) if row else None
    except (TypeError, ValueError):
        return None


def fts5_index_is_current(
    db_path: str, *, conn: sqlite3.Connection | None = None
) -> bool:
    """判断 FTS5 数据库是否使用当前要求的索引架构版本。"""
    return fts5_index_version(db_path, conn=conn) == index_schema.CURRENT_INDEX_SCHEMA_VERSION


def _parent_chunk_rows(splits: list) -> list[tuple[str, str, str, str, str]]:
    """从内存中的父子切分结果收集去重后的父段落记录。"""
    parents: dict[str, tuple[str, str, str, str, str]] = {}
    for document in splits:
        parent_id = document.metadata.get("parent_id")
        parent_content = document.metadata.get("parent_content")
        if not parent_id or not parent_content:
            continue
        parent_id = str(parent_id)
        parents[parent_id] = (
            parent_id,
            str(document.metadata.get("source", "")),
            str(document.metadata.get("page", document.metadata.get("row", ""))),
            str(parent_content),
            str(document.metadata.get("contextual_title", "")),
        )
    return list(parents.values())


def _fts5_document_rows(splits: list, source_filename: str = "") -> list[tuple]:
    """将检索切片转换为可批量写入 FTS5 docs 表的行。"""
    return [
        (
            " ".join(jieba.cut(contextualized_chunk_text(document))),
            document.metadata.get("source", source_filename),
            str(document.metadata.get("page", document.metadata.get("row", ""))),
            document.page_content,
            int(document.metadata.get("chunk_index", 0)),
            document.metadata.get("parent_id", ""),
            document.metadata.get("contextual_title", ""),
            chunk_route(document),
        )
        for document in splits
    ]


def build_fts5_index(splits: list, db_path: str) -> None:
    """为一个知识库构建完整 FTS5 索引。

    数据库先在同目录临时路径中构建，再原子替换正式文件，因此重建失败仍保留此前
    可检索的索引。
    """
    splits = assign_chunk_indexes(splits)
    parent_rows = _parent_chunk_rows(splits)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    temporary_path = f"{db_path}.{uuid.uuid4().hex}.tmp"
    try:
        with application_db.connect_fts5(temporary_path) as conn:
            conn.execute("PRAGMA secure_delete = ON")
            conn.execute(
                f"PRAGMA user_version = {index_schema.CURRENT_INDEX_SCHEMA_VERSION}"
            )
            conn.execute(
                f"CREATE TABLE {PARENT_CHUNK_TABLE} ("
                "parent_id TEXT PRIMARY KEY, source TEXT NOT NULL, page TEXT NOT NULL, "
                "content TEXT NOT NULL, contextual_title TEXT NOT NULL DEFAULT '')"
            )
            conn.execute(
                "CREATE VIRTUAL TABLE docs USING fts5("
                "content, source UNINDEXED, page UNINDEXED, original_text UNINDEXED, "
                "chunk_index UNINDEXED, parent_id UNINDEXED, contextual_title UNINDEXED, "
                "chunk_route UNINDEXED, "
                "tokenize='unicode61 remove_diacritics 2')"
            )
            batch = []
            for row in _fts5_document_rows(splits):
                batch.append(row)
                if len(batch) >= 1000:
                    conn.executemany("INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
            if batch:
                conn.executemany("INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
            if parent_rows:
                conn.executemany(
                    f"INSERT INTO {PARENT_CHUNK_TABLE} "
                    "(parent_id, source, page, content, contextual_title) VALUES (?, ?, ?, ?, ?)",
                    parent_rows,
                )
        os.replace(temporary_path, db_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def update_fts5_index(
    kb_name: str,
    source_filename: str,
    splits: list,
) -> None:
    """将一个文件的切片更新到已有 FTS5 索引中。"""
    splits = assign_chunk_indexes(splits)
    db_path = fts_db_path(kb_name)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if not os.path.exists(db_path):
        build_fts5_index(splits, db_path)
        return
    with application_db.connect_fts5(db_path) as conn:
        if not fts5_index_is_current(db_path, conn=conn):
            raise RuntimeError("FTS5 index schema is outdated and must be rebuilt")
        conn.execute("PRAGMA secure_delete = ON")
        conn.execute("DELETE FROM docs WHERE source = ?", (source_filename,))
        conn.execute(f"DELETE FROM {PARENT_CHUNK_TABLE} WHERE source = ?", (source_filename,))
        rows = _fts5_document_rows(splits, source_filename)
        if rows:
            conn.executemany("INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        parent_rows = _parent_chunk_rows(splits)
        if parent_rows:
            conn.executemany(
                f"INSERT INTO {PARENT_CHUNK_TABLE} "
                "(parent_id, source, page, content, contextual_title) VALUES (?, ?, ?, ?, ?)",
                parent_rows,
            )


def delete_from_fts5(
    kb_name: str,
    source_filename: str,
) -> None:
    """从 FTS5 索引删除一个文件的切片和父段落记录。"""
    db_path = fts_db_path(kb_name)
    if not os.path.exists(db_path):
        return
    with application_db.connect_fts5(db_path) as conn:
        if not fts5_index_is_current(db_path, conn=conn):
            raise RuntimeError("FTS5 index schema is outdated and must be rebuilt")
        conn.execute("PRAGMA secure_delete = ON")
        conn.execute("DELETE FROM docs WHERE source = ?", (source_filename,))
        conn.execute(f"DELETE FROM {PARENT_CHUNK_TABLE} WHERE source = ?", (source_filename,))


def remove_empty_fts_index(
    kb_name: str, has_documents: Callable[[str], bool]
) -> bool:
    """知识库不再有源文件时，删除其 FTS 索引目录。"""
    if has_documents(kb_name):
        return False
    index_dir = vector_store_dir(kb_name)
    if not os.path.isdir(index_dir):
        return False
    shutil.rmtree(index_dir)
    return True


def remove_fts_index(kb_name: str) -> bool:
    """知识库自身被删除时，清除其全部 FTS 存储。"""
    index_dir = vector_store_dir(kb_name)
    if not os.path.isdir(index_dir):
        return False
    shutil.rmtree(index_dir)
    return True


def fts5_available_routes(
    db_path: str,
    allowed_sources: list[str] | None,
) -> set[str] | None:
    """返回匹配的分块路线；索引缺失或过期时返回 None。"""
    if not os.path.exists(db_path) or allowed_sources == []:
        return set() if allowed_sources == [] else None
    statement = "SELECT DISTINCT chunk_route FROM docs"
    params: list[str] = []
    if allowed_sources is not None:
        placeholders = ",".join("?" for _ in allowed_sources)
        statement += f" WHERE source IN ({placeholders})"
        params.extend(allowed_sources)
    try:
        with application_db.connect_fts5(db_path) as conn:
            if not fts5_index_is_current(db_path, conn=conn):
                return None
            return {
                str(row[0] or CHUNK_ROUTE_FIXED)
                for row in conn.execute(statement, params).fetchall()
            }
    except sqlite3.Error as exc:
        logger.warning("FTS5 route availability check failed: %s", exc)
        return set()
