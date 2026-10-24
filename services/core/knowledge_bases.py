"""Relational persistence for knowledge-base lifecycle and dashboard data."""

import json
from collections.abc import Callable
from typing import cast

from services.core import sqlite_connections as application_db
from services.core.contracts import (
    DashboardDocumentRow,
    DashboardTrendRow,
    KnowledgeBaseSummary,
)


def _summary_from_row(row: tuple[object, ...]) -> KnowledgeBaseSummary:
    """把知识库聚合查询结果转换为列表摘要。"""
    created_at = str(row[1]) if row[1] else ""
    return {
        "name": str(row[0]),
        "date": created_at[:10],
        "docs": cast(int, row[2]),
    }


def _dashboard_document_from_row(row: tuple[object, ...]) -> DashboardDocumentRow:
    """把仪表盘文档查询结果转换为带类型的文档记录。"""
    return DashboardDocumentRow(
        kb_name=str(row[0]),
        filename=str(row[1]) if row[1] is not None else None,
        size=cast(int, row[2]) if row[2] is not None else None,
        status=str(row[3]) if row[3] is not None else None,
    )


def _dashboard_trend_from_row(row: tuple[object, ...]) -> DashboardTrendRow:
    """把按日期聚合的会话和回答数量转换为趋势记录。"""
    return DashboardTrendRow(
        date=str(row[0]),
        session_count=cast(int, row[1] or 0),
        answer_count=cast(int, row[2] or 0),
    )


class KnowledgeBaseStore:
    """Persist knowledge-base lifecycle state and cross-table references."""

    def __init__(
        self,
        db_file: Callable[[], str],
        now: Callable[[], str],
        validate_kb_name: Callable[[str], str],
    ) -> None:
        """保存延迟解析的数据库路径、时钟和名称校验依赖。"""
        self._db_file = db_file
        self._now = now
        self._validate_kb_name = validate_kb_name

    def exists(self, kb_name: str) -> bool:
        """判断通过校验的知识库是否存在。"""
        kb_name = self._validate_kb_name(kb_name)
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT 1 FROM knowledge_bases WHERE name = ?", (kb_name,)
            ).fetchone()
        return row is not None

    def list_summaries(self) -> list[KnowledgeBaseSummary]:
        """返回知识库列表及其创建时间和文档数量。"""
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                "SELECT kb.name, kb.created_at, COUNT(doc.id) "
                "FROM knowledge_bases kb "
                "LEFT JOIN documents doc ON kb.name = doc.kb_name "
                "GROUP BY kb.name"
            ).fetchall()
        return [_summary_from_row(row) for row in rows]

    def create(self, kb_name: str, chunk_size: int, chunk_overlap: int) -> None:
        """创建知识库并写入初始分块配置。"""
        kb_name = self._validate_kb_name(kb_name)
        with application_db.connect(self._db_file()) as conn:
            conn.execute(
                "INSERT INTO knowledge_bases "
                "(name, created_at, chunk_size, chunk_overlap) VALUES (?, ?, ?, ?)",
                (kb_name, self._now(), chunk_size, chunk_overlap),
            )

    @staticmethod
    def _rewrite_chat_sources(conn, old_name: str, new_name: str | None) -> None:
        """重命名或移除已保存引用中的知识库名称。"""
        rows = conn.execute(
            "SELECT id, sources FROM chat_messages "
            "WHERE EXISTS ("
            "SELECT 1 FROM json_each("
            "CASE WHEN json_valid(chat_messages.sources) THEN chat_messages.sources ELSE '[]' END"
            ") AS source WHERE json_extract(source.value, '$.kb_name') = ?)",
            (old_name,),
        ).fetchall()
        for message_id, raw_sources in rows:
            try:
                sources = json.loads(raw_sources)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(sources, list):
                continue
            changed = False
            rewritten = []
            for source in sources:
                source_kb_name = source.get("kb_name") if isinstance(source, dict) else None
                if isinstance(source, dict) and source_kb_name == old_name:
                    changed = True
                    if new_name is None:
                        continue
                    source = {**source, "kb_name": new_name}
                rewritten.append(source)
            if changed:
                conn.execute(
                    "UPDATE chat_messages SET sources = ? WHERE id = ?",
                    (json.dumps(rewritten, ensure_ascii=False), message_id),
                )

    @staticmethod
    def _rewrite_role_permissions(conn, old_name: str, new_name: str | None) -> None:
        """在所有角色权限列表中重命名或移除知识库。"""
        rows = conn.execute(
            "SELECT id, knowledge_base_permissions FROM roles "
            "WHERE EXISTS ("
            "SELECT 1 FROM json_each("
            "CASE WHEN json_valid(roles.knowledge_base_permissions) "
            "THEN roles.knowledge_base_permissions ELSE '[]' END"
            ") AS permission WHERE permission.value = ?)",
            (old_name,),
        ).fetchall()
        for role_id, raw_permissions in rows:
            try:
                permissions = json.loads(raw_permissions or "[]")
            except json.JSONDecodeError:
                continue
            if not isinstance(permissions, list) or old_name not in permissions:
                continue
            rewritten = [new_name if item == old_name else item for item in permissions]
            rewritten = list(dict.fromkeys(item for item in rewritten if item))
            conn.execute(
                "UPDATE roles SET knowledge_base_permissions = ? WHERE id = ?",
                (json.dumps(rewritten, ensure_ascii=False), role_id),
            )

    def delete_relational_data(self, kb_name: str) -> None:
        """在一个事务中删除知识库拥有的全部主数据库记录。"""
        kb_name = self._validate_kb_name(kb_name)
        with application_db.connect(self._db_file()) as conn:
            conn.execute(
                "DELETE FROM chat_session_memory WHERE session_id IN "
                "(SELECT id FROM chat_sessions WHERE kb_name = ?)",
                (kb_name,),
            )
            conn.execute(
                "DELETE FROM chat_messages WHERE session_id IN "
                "(SELECT id FROM chat_sessions WHERE kb_name = ?)",
                (kb_name,),
            )
            conn.execute("DELETE FROM chat_sessions WHERE kb_name = ?", (kb_name,))
            self._rewrite_chat_sources(conn, kb_name, None)
            self._rewrite_role_permissions(conn, kb_name, None)
            conn.execute(
                "DELETE FROM chroma_disabled_sources WHERE kb_name = ?", (kb_name,)
            )
            conn.execute("DELETE FROM knowledge_bases WHERE name = ?", (kb_name,))
            conn.execute("DELETE FROM documents WHERE kb_name = ?", (kb_name,))

    def rename_relational_data(
        self, old_name: str, new_name: str, old_path: str, new_path: str
    ) -> None:
        """外部文件迁移完成后，统一改写主数据库中的知识库引用。"""
        old_name = self._validate_kb_name(old_name)
        new_name = self._validate_kb_name(new_name)
        with application_db.connect(self._db_file()) as conn:
            conn.execute(
                "UPDATE knowledge_bases SET name = ?, active_rebuild_job_id = '' "
                "WHERE name = ?",
                (new_name, old_name),
            )
            conn.execute(
                "UPDATE documents SET kb_name = ?, path = REPLACE(path, ?, ?) "
                "WHERE kb_name = ?",
                (new_name, old_path, new_path, old_name),
            )
            conn.execute(
                "UPDATE index_jobs SET kb_name = ? WHERE kb_name = ?",
                (new_name, old_name),
            )
            conn.execute(
                "DELETE FROM chroma_disabled_sources WHERE kb_name = ?", (new_name,)
            )
            conn.execute(
                "UPDATE chroma_disabled_sources SET kb_name = ? WHERE kb_name = ?",
                (new_name, old_name),
            )
            conn.execute(
                "UPDATE chat_sessions "
                "SET kb_name = ?, title = CASE WHEN title = ? THEN ? ELSE title END "
                "WHERE kb_name = ?",
                (new_name, f"{old_name} 会话", f"{new_name} 会话", old_name),
            )
            self._rewrite_chat_sources(conn, old_name, new_name)
            self._rewrite_role_permissions(conn, old_name, new_name)

    def dashboard_documents(
        self, allowed_kbs: tuple[str, ...] | None
    ) -> list[DashboardDocumentRow]:
        """返回用于汇总仪表盘状态和容量的数据行。"""
        if allowed_kbs is not None and not allowed_kbs:
            return []
        statement = (
            "SELECT kb.name, doc.filename, doc.size, doc.status "
            "FROM knowledge_bases kb LEFT JOIN documents doc ON doc.kb_name = kb.name"
        )
        params: tuple[str, ...] = ()
        if allowed_kbs is not None:
            placeholders = ",".join("?" for _ in allowed_kbs)
            statement += f" WHERE kb.name IN ({placeholders})"
            params = allowed_kbs
        statement += " ORDER BY doc.id"
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(statement, params).fetchall()
        return [_dashboard_document_from_row(row) for row in rows]

    def dashboard_trend(
        self,
        allowed_kbs: tuple[str, ...] | None,
        visible_kbs: list[str],
        *,
        days: int,
    ) -> list[DashboardTrendRow]:
        """返回仪表盘时间窗口内每天的会话数和回答数。"""
        window = f"DATE('now', '-{int(days)} days')"
        with application_db.connect(self._db_file()) as conn:
            if allowed_kbs is None:
                rows = conn.execute(
                    f"SELECT DATE(created_at), "
                    "SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END), "
                    "SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) "
                    f"FROM chat_messages WHERE created_at >= {window} "
                    "GROUP BY DATE(created_at) ORDER BY DATE(created_at)"
                ).fetchall()
                return [_dashboard_trend_from_row(row) for row in rows]
            if not visible_kbs:
                return []
            placeholders = ",".join("?" for _ in visible_kbs)
            rows = conn.execute(
                "SELECT DATE(m.created_at), "
                "SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) "
                "FROM chat_messages m JOIN chat_sessions s ON s.id = m.session_id "
                f"WHERE m.created_at >= {window} AND s.kb_name IN ({placeholders}) "
                "GROUP BY DATE(m.created_at) ORDER BY DATE(m.created_at)",
                visible_kbs,
            ).fetchall()
            return [_dashboard_trend_from_row(row) for row in rows]
