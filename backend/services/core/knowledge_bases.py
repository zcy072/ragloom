"""Relational persistence for knowledge-base lifecycle and dashboard data."""

import json
import sqlite3
import uuid
from collections.abc import Callable
from typing import cast

from services.chat import source_access
from services.core import sqlite_connections as application_db
from services.core.contracts import (
    DashboardDocumentRow,
    DashboardTrendRow,
    KnowledgeBase,
    KnowledgeBaseSummary,
)

COMMON_SYSTEM_KEY = "common"


class KnowledgeBaseNameConflict(ValueError):
    """知识库显示名称已被占用（包括正在删除、尚未清理完的库）。"""


def _knowledge_base_from_row(row: tuple[object, ...]) -> KnowledgeBase:
    """把 knowledge_bases 的 (kb_id, name, system_key, state, created_at) 行转换为记录。"""
    return KnowledgeBase(
        kb_id=str(row[0]),
        name=str(row[1]),
        system_key=application_db.optional_text(row[2]),
        state=str(row[3]),
        created_at=str(row[4]),
    )


def _summary_from_row(row: tuple[object, ...]) -> KnowledgeBaseSummary:
    """把知识库聚合查询结果转换为列表摘要。"""
    created_at = str(row[4]) if row[4] else ""
    return {
        "kb_id": str(row[0]),
        "name": str(row[1]),
        "system_key": application_db.optional_text(row[2]),
        "state": str(row[3]),
        "date": created_at[:10],
        "docs": cast(int, row[5]),
    }


def _dashboard_document_from_row(row: tuple[object, ...]) -> DashboardDocumentRow:
    """把仪表盘文档查询结果转换为带类型的文档记录。"""
    return DashboardDocumentRow(
        kb_id=str(row[0]),
        kb_name=str(row[1]),
        filename=str(row[2]) if row[2] is not None else None,
        size=cast(int, row[3]) if row[3] is not None else None,
        status=str(row[4]) if row[4] is not None else None,
    )


def _dashboard_trend_from_row(row: tuple[object, ...]) -> DashboardTrendRow:
    """把按日期聚合的提问和回答数量转换为趋势记录。"""
    return DashboardTrendRow(
        date=str(row[0]),
        question_count=cast(int, row[1] or 0),
        answer_count=cast(int, row[2] or 0),
    )


class KnowledgeBaseStore:
    """Persist knowledge-base lifecycle state; every lookup is by `kb_id`."""

    def __init__(
        self,
        db_file: Callable[[], str],
        now: Callable[[], str],
        validate_kb_name: Callable[[str], str],
        validate_kb_id: Callable[[str], str],
    ) -> None:
        """保存延迟解析的数据库路径、时钟和名称、ID 校验依赖。"""
        self._db_file = db_file
        self._now = now
        self._validate_kb_name = validate_kb_name
        self._validate_kb_id = validate_kb_id

    def get(self, kb_id: str) -> KnowledgeBase | None:
        """按 ID 返回知识库记录（含 deleting 状态），不存在时返回 None。"""
        kb_id = self._validate_kb_id(kb_id)
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT kb_id, name, system_key, state, created_at "
                "FROM knowledge_bases WHERE kb_id = ?",
                (kb_id,),
            ).fetchone()
        return _knowledge_base_from_row(row) if row else None

    def exists(self, kb_id: str) -> bool:
        """判断知识库是否存在且处于 active 状态。"""
        kb_id = self._validate_kb_id(kb_id)
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT 1 FROM knowledge_bases WHERE kb_id = ? AND state = 'active'",
                (kb_id,),
            ).fetchone()
        return row is not None

    def common_kb_id(self) -> str | None:
        """返回通用知识库的 ID；它由 system_key 识别，与显示名称无关。"""
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT kb_id FROM knowledge_bases WHERE system_key = ? AND state = 'active'",
                (COMMON_SYSTEM_KEY,),
            ).fetchone()
        return str(row[0]) if row else None

    def names_for(self, kb_ids: list[str]) -> dict[str, str]:
        """批量取当前显示名称，仅用于响应展示；已删除的 ID 不出现在结果里。"""
        if not kb_ids:
            return {}
        placeholders = ",".join("?" for _ in kb_ids)
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                f"SELECT kb_id, name FROM knowledge_bases WHERE kb_id IN ({placeholders})",
                tuple(kb_ids),
            ).fetchall()
        return {str(kb_id): str(name) for kb_id, name in rows}

    def list_summaries(self, *, include_deleting: bool = False) -> list[KnowledgeBaseSummary]:
        """返回知识库列表及其创建时间和文档数量；默认只含 active。"""
        statement = (
            "SELECT kb.kb_id, kb.name, kb.system_key, kb.state, kb.created_at, "
            "COUNT(doc.document_id) FROM knowledge_bases kb "
            "LEFT JOIN documents doc ON kb.kb_id = doc.kb_id "
        )
        if not include_deleting:
            statement += "WHERE kb.state = 'active' "
        statement += "GROUP BY kb.kb_id"
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(statement).fetchall()
        return [_summary_from_row(row) for row in rows]

    def create(
        self,
        name: str,
        chunk_size: int,
        chunk_overlap: int,
        *,
        system_key: str | None = None,
        grant_role_id: int | None = None,
    ) -> KnowledgeBase:
        """创建知识库，生成不可变的 kb_id；名称已被占用时抛出 KnowledgeBaseNameConflict。

        grant_role_id 不为空时，在同一事务里把新知识库授予该角色。
        """
        name = self._validate_kb_name(name)
        record = KnowledgeBase(str(uuid.uuid4()), name, system_key, "active", self._now())
        try:
            with application_db.connect(self._db_file()) as conn:
                conn.execute(
                    "INSERT INTO knowledge_bases "
                    "(kb_id, name, system_key, state, created_at, chunk_size, chunk_overlap) "
                    "VALUES (?, ?, ?, 'active', ?, ?, ?)",
                    (
                        record.kb_id,
                        record.name,
                        record.system_key,
                        record.created_at,
                        chunk_size,
                        chunk_overlap,
                    ),
                )
                if grant_role_id is not None:
                    conn.execute(
                        "INSERT INTO role_knowledge_bases (role_id, kb_id) VALUES (?, ?)",
                        (grant_role_id, record.kb_id),
                    )
        except sqlite3.IntegrityError as exc:
            raise KnowledgeBaseNameConflict("知识库名称已存在") from exc
        return record

    def rename(self, kb_id: str, new_name: str) -> KnowledgeBase:
        """只更新显示名称：一个事务、一行 UPDATE，不触碰文件、任务、权限或索引。

        不存在或不处于 active 状态时抛出 LookupError，名称冲突时抛出
        KnowledgeBaseNameConflict。
        """
        kb_id = self._validate_kb_id(kb_id)
        new_name = self._validate_kb_name(new_name)
        try:
            with application_db.connect(self._db_file()) as conn:
                cursor = conn.execute(
                    "UPDATE knowledge_bases SET name = ? WHERE kb_id = ? AND state = 'active'",
                    (new_name, kb_id),
                )
                if cursor.rowcount != 1:
                    raise LookupError("知识库不存在")
                row = conn.execute(
                    "SELECT kb_id, name, system_key, state, created_at "
                    "FROM knowledge_bases WHERE kb_id = ?",
                    (kb_id,),
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise KnowledgeBaseNameConflict("知识库名称已存在") from exc
        return _knowledge_base_from_row(row)

    def mark_deleting(self, kb_id: str) -> KnowledgeBase | None:
        """把 active 的库改为 deleting 并清空重建指针；已在 deleting 则幂等返回。

        不存在时返回 None。调用方必须已持有该 kb_id 的整库锁。
        """
        kb_id = self._validate_kb_id(kb_id)
        with application_db.connect(self._db_file()) as conn:
            conn.execute(
                "UPDATE knowledge_bases SET state = 'deleting', "
                "active_rebuild_job_id = NULL WHERE kb_id = ?",
                (kb_id,),
            )
            row = conn.execute(
                "SELECT kb_id, name, system_key, state, created_at "
                "FROM knowledge_bases WHERE kb_id = ?",
                (kb_id,),
            ).fetchone()
        return _knowledge_base_from_row(row) if row else None

    def deleting_ids(self) -> list[str]:
        """返回上次未完成删除的知识库 ID，供启动时重试清理。"""
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                "SELECT kb_id FROM knowledge_bases WHERE state = 'deleting'"
            ).fetchall()
        return [str(row[0]) for row in rows]

    @staticmethod
    def _strip_chat_sources(conn: sqlite3.Connection, kb_id: str) -> None:
        """从其他会话已保存的引用中移除指向该库的顶层来源和嵌套 evidence。

        只按结构化的 kb_id 字段匹配，不触碰回答正文。
        """
        rows = conn.execute(
            "SELECT id, sources FROM chat_messages "
            "WHERE EXISTS ("
            "SELECT 1 FROM json_each("
            "CASE WHEN json_valid(chat_messages.sources) THEN chat_messages.sources ELSE '[]' END"
            ") AS source WHERE json_extract(source.value, '$.kb_id') = ? "
            "OR EXISTS (SELECT 1 FROM json_each(source.value, '$.evidence') AS ev "
            "WHERE json_extract(ev.value, '$.kb_id') = ?))",
            (kb_id, kb_id),
        ).fetchall()
        for message_id, raw_sources in rows:
            try:
                sources = json.loads(raw_sources)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(sources, list):
                continue
            kept = []
            for source in sources:
                if not isinstance(source, dict):
                    kept.append(source)
                    continue
                if source.get("kb_id") == kb_id:
                    continue
                evidence = source.get("evidence")
                if isinstance(evidence, list):
                    source = {
                        **source,
                        "evidence": [
                            item
                            for item in evidence
                            if not (isinstance(item, dict) and item.get("kb_id") == kb_id)
                        ],
                    }
                kept.append(source)
            conn.execute(
                "UPDATE chat_messages SET sources = ? WHERE id = ?",
                (json.dumps(kept, ensure_ascii=False), message_id),
            )

    def delete_relational_data(self, kb_id: str) -> None:
        """外部资源清理完成后，一个事务内删除知识库行；其余记录由外键级联清理。"""
        kb_id = self._validate_kb_id(kb_id)
        with application_db.connect(self._db_file()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            traces = conn.execute(
                "SELECT id, payload FROM chat_traces WHERE EXISTS "
                "(SELECT 1 FROM json_tree(CASE WHEN json_valid(payload) THEN payload ELSE '{}' END) "
                "WHERE key = 'kb_id' AND value = ?)",
                (kb_id,),
            ).fetchall()
            for trace_id, raw_payload in traces:
                payload = json.loads(raw_payload)
                filtered = source_access.filter_trace_payload(
                    payload, lambda source_id: source_id != kb_id
                )
                conn.execute(
                    "UPDATE chat_traces SET payload = ? WHERE id = ?",
                    (json.dumps(filtered, ensure_ascii=False), trace_id),
                )
            self._strip_chat_sources(conn, kb_id)
            conn.execute("DELETE FROM knowledge_bases WHERE kb_id = ?", (kb_id,))

    def dashboard_documents(
        self, allowed_kb_ids: tuple[str, ...] | None
    ) -> list[DashboardDocumentRow]:
        """返回用于汇总仪表盘状态和容量的数据行。"""
        if allowed_kb_ids is not None and not allowed_kb_ids:
            return []
        statement = (
            "SELECT kb.kb_id, kb.name, doc.filename, doc.size, doc.status "
            "FROM knowledge_bases kb LEFT JOIN documents doc ON doc.kb_id = kb.kb_id "
            "WHERE kb.state = 'active'"
        )
        params: tuple[str, ...] = ()
        if allowed_kb_ids is not None:
            placeholders = ",".join("?" for _ in allowed_kb_ids)
            statement += f" AND kb.kb_id IN ({placeholders})"
            params = allowed_kb_ids
        statement += " ORDER BY kb.rowid, doc.uploaded_at, doc.document_id"
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(statement, params).fetchall()
        return [_dashboard_document_from_row(row) for row in rows]

    def dashboard_trend(
        self,
        allowed_kb_ids: tuple[str, ...] | None,
        visible_kb_ids: list[str],
        *,
        days: int,
    ) -> list[DashboardTrendRow]:
        """返回仪表盘时间窗口内每天的提问数和回答数。

        时间以 UTC 保存，按本地日期分组，与用户分析一致。
        """
        window = (
            f"strftime('%Y-%m-%dT%H:%M:%S', DATE('now', 'localtime', '-{int(days)} days'), 'utc')"
        )
        with application_db.connect(self._db_file()) as conn:
            if allowed_kb_ids is None:
                rows = conn.execute(
                    "SELECT DATE(created_at, 'localtime'), "
                    "SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END), "
                    "SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) "
                    f"FROM chat_messages WHERE created_at >= {window} "
                    "GROUP BY 1 ORDER BY 1"
                ).fetchall()
                return [_dashboard_trend_from_row(row) for row in rows]
            if not visible_kb_ids:
                return []
            placeholders = ",".join("?" for _ in visible_kb_ids)
            rows = conn.execute(
                "SELECT DATE(m.created_at, 'localtime'), "
                "SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) "
                "FROM chat_messages m JOIN chat_sessions s ON s.id = m.session_id "
                f"WHERE m.created_at >= {window} AND s.kb_id IN ({placeholders}) "
                "GROUP BY 1 ORDER BY 1",
                visible_kb_ids,
            ).fetchall()
            return [_dashboard_trend_from_row(row) for row in rows]
