import asyncio
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

import app_storage
from services.chat import memory as chat_memory
from services.chat import sessions as chat_sessions
from services.chat.sessions import SessionOwner
from services.core import config as application_config
from tests.support import IsolatedStorageTestCase


class ChatMemoryTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO knowledge_bases (name, kb_id, created_at) VALUES (?, lower(hex(randomblob(16))), ?)",
                ("技术部", application_config.db_now()),
            )
        self.session_id = chat_sessions.resolve_session(
            None, SessionOwner(1, "alice"), "技术部"
        ).session_id

    def _insert_messages(self, count: int) -> None:
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.executemany(
                "INSERT INTO chat_messages (session_id, role, content, sources, created_at) "
                "VALUES (?, ?, ?, '[]', ?)",
                [
                    (
                        self.session_id,
                        "user" if index % 2 == 0 else "assistant",
                        f"第 {index + 1} 条消息",
                        application_config.db_now(),
                    )
                    for index in range(count)
                ],
            )

    def test_summary_refresh_compacts_older_messages_and_keeps_recent_originals(self):
        self._insert_messages(14)

        class FakeSummaryModel:
            def __init__(self):
                self.messages = []

            async def ainvoke(self, messages):
                self.messages = messages
                return SimpleNamespace(content="- 用户正在持续讨论技术部制度。")

        model = FakeSummaryModel()
        with patch.object(chat_memory, "create_summary_llm", return_value=model):
            refreshed = asyncio.run(
                chat_memory.refresh_conversation_summary(
                    self.session_id,
                    {
                        "api_key": "test-key",
                        "base_url": "https://example.test/v1",
                        "model": "test-model",
                    },
                )
            )

        self.assertTrue(refreshed)
        memory = chat_sessions.load_conversation_memory(
            self.session_id,
            application_config.CHAT_MEMORY_KEEP_RECENT_MESSAGE_COUNT,
            application_config.CHAT_MEMORY_RECENT_MAX_CHARS,
        )
        self.assertEqual(memory.summary, "- 用户正在持续讨论技术部制度。")
        self.assertEqual(len(memory.recent_history), 8)
        self.assertEqual(memory.recent_history[0][1], "第 14 条消息")
        self.assertEqual(memory.recent_history[-1][1], "第 7 条消息")
        self.assertIn("第 1 条消息", model.messages[-1].content)
        self.assertIn("第 6 条消息", model.messages[-1].content)
        self.assertNotIn("第 7 条消息", model.messages[-1].content)

    def test_summary_refresh_below_threshold_does_not_create_model_client(self):
        self._insert_messages(
            application_config.CHAT_MEMORY_SUMMARY_TRIGGER_MESSAGE_COUNT
        )

        with patch.object(chat_memory, "create_summary_llm") as model_factory:
            refreshed = asyncio.run(
                chat_memory.refresh_conversation_summary(
                    self.session_id,
                    {
                        "api_key": "test-key",
                        "base_url": "https://example.test/v1",
                        "model": "test-model",
                    },
                )
            )

        self.assertFalse(refreshed)
        model_factory.assert_not_called()

    def test_stale_summary_update_cannot_overwrite_newer_memory(self):
        self._insert_messages(14)
        batch = chat_sessions.load_summary_batch(self.session_id, 12, 8)
        self.assertIsNotNone(batch)
        first_message_id = batch.messages[-1][0]
        self.assertTrue(
            chat_sessions.save_conversation_summary(
                self.session_id,
                "较新的摘要",
                expected_message_id=0,
                through_message_id=first_message_id,
            )
        )
        self.assertFalse(
            chat_sessions.save_conversation_summary(
                self.session_id,
                "过期摘要",
                expected_message_id=0,
                through_message_id=first_message_id,
            )
        )
        memory = chat_sessions.load_conversation_memory(self.session_id, 8, 6000)
        self.assertEqual(memory.summary, "较新的摘要")

    def test_recent_history_respects_character_budget_from_newest_message(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.executemany(
                "INSERT INTO chat_messages (session_id, role, content, sources, created_at) "
                "VALUES (?, 'user', ?, '[]', ?)",
                [
                    (self.session_id, "较早消息", application_config.db_now()),
                    (self.session_id, "最新消息内容", application_config.db_now()),
                ],
            )

        memory = chat_sessions.load_conversation_memory(self.session_id, 12, 6)

        self.assertEqual(memory.recent_history, (("user", "最新消息内容"),))

    def test_unsummarized_backlog_temporarily_expands_recent_message_window(self):
        """摘要失败积压时仍应在字符预算内读取超过常规上限的原文。"""
        self._insert_messages(14)

        memory = chat_sessions.load_conversation_memory(self.session_id, 12, 6000)

        self.assertEqual(len(memory.recent_history), 14)
        self.assertEqual(memory.recent_history[-1][1], "第 1 条消息")

    def test_recent_history_keeps_user_question_when_latest_turn_exceeds_budget(self):
        """最新问答轮过长时优先保留用户问题，避免只剩助手回答。"""
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.executemany(
                "INSERT INTO chat_messages (session_id, role, content, sources, created_at) "
                "VALUES (?, ?, ?, '[]', ?)",
                [
                    (self.session_id, "user", "用户问题", application_config.db_now()),
                    (self.session_id, "assistant", "很长的助手回答", application_config.db_now()),
                ],
            )

        memory = chat_sessions.load_conversation_memory(self.session_id, 12, 6)

        self.assertEqual(memory.recent_history[-1], ("user", "用户问题"))
        self.assertEqual(sum(len(text) for _, text in memory.recent_history), 6)

    def test_oversized_summary_message_keeps_both_ends_before_advancing_boundary(self):
        """单条超长消息进入摘要时应保留首尾，而不是永久丢掉尾部。"""
        message = ((17, "user", "开头信息" + "中" * 80 + "结尾信息"),)

        with patch.object(
            application_config, "CHAT_MEMORY_SUMMARY_MAX_INPUT_CHARS", 40
        ):
            transcript, through_message_id = chat_memory._bounded_transcript(message)

        self.assertEqual(through_message_id, 17)
        self.assertIn("开头信息", transcript)
        self.assertIn("结尾信息", transcript)
        self.assertIn("中间内容因长度限制省略", transcript)

    def test_summary_failure_is_visible_in_warning_log(self):
        """摘要模型异常应留下可见告警，同时不影响主回答。"""
        self._insert_messages(14)

        class FailingSummaryModel:
            async def ainvoke(self, messages):
                raise RuntimeError("摘要服务不可用")

        with patch.object(
            chat_memory, "create_summary_llm", return_value=FailingSummaryModel()
        ), self.assertLogs(chat_memory.logger, level="WARNING") as logs:
            refreshed = asyncio.run(
                chat_memory.refresh_conversation_summary(
                    self.session_id,
                    {"api_key": "key", "base_url": "url", "model": "model"},
                )
            )

        self.assertFalse(refreshed)
        self.assertIn("摘要服务不可用", "\n".join(logs.output))

    def test_summary_cannot_recreate_memory_after_session_deletion(self):
        deleted_session_id = self.session_id
        chat_sessions.delete_owned_session(
            deleted_session_id, SessionOwner(1, "alice")
        )

        saved = chat_sessions.save_conversation_summary(
            deleted_session_id,
            "不应保存的摘要",
            expected_message_id=0,
            through_message_id=1,
        )

        self.assertFalse(saved)
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            memory_count = conn.execute(
                "SELECT COUNT(*) FROM chat_session_memory WHERE session_id = ?",
                (deleted_session_id,),
            ).fetchone()[0]
        self.assertEqual(memory_count, 0)

    def _insert_turns(self, turns: list[tuple[str, str]]) -> None:
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.executemany(
                "INSERT INTO chat_messages (session_id, role, content, sources, created_at) "
                "VALUES (?, ?, ?, '[]', ?)",
                [
                    (self.session_id, role, content, application_config.db_now())
                    for role, content in turns
                ],
            )

    def _insert_long_turns(self, count: int) -> None:
        """插入 count 轮问答，回答长度等于单条上限；每轮约 1503 字，6000 字预算只放得下 3 轮。"""
        long_answer = "答" * application_config.CHAT_MEMORY_RECENT_MESSAGE_MAX_CHARS
        self._insert_turns(
            [
                (role, text)
                for index in range(count)
                for role, text in (
                    ("user", f"问题{index + 1}"),
                    ("assistant", long_answer),
                )
            ]
        )

    def test_messages_pushed_out_of_recent_window_are_summarized_before_threshold(self):
        """长回答让近期窗口放不下较早的轮次时，这些轮次要提前并入摘要。"""
        # 5 轮 = 10 条，未超过 12 条的触发阈值。
        self._insert_long_turns(5)

        batch = chat_sessions.load_summary_batch(self.session_id, 12, 8)

        self.assertIsNotNone(batch)
        questions = [c for _, role, c in batch.messages if role == "user"]
        self.assertEqual(questions, ["问题1", "问题2"])

    def test_no_early_summary_when_recent_window_holds_everything(self):
        self._insert_messages(8)

        self.assertIsNone(chat_sessions.load_summary_batch(self.session_id, 12, 8))

    def test_summary_batch_ends_exactly_where_recent_window_begins(self):
        """摘要吸收的消息与近期原文窗口首尾相接，不重叠也不留缺口。"""
        self._insert_long_turns(5)
        batch = chat_sessions.load_summary_batch(self.session_id, 12, 8)
        self.assertIsNotNone(batch)
        chat_sessions.save_conversation_summary(
            self.session_id,
            "摘要",
            expected_message_id=0,
            through_message_id=batch.messages[-1][0],
        )

        memory = chat_sessions.load_conversation_memory(self.session_id, 12, 6000)

        summarized = [c for _, role, c in batch.messages if role == "user"]
        recent = [c for role, c in memory.recent_history if role == "user"]
        self.assertEqual(summarized + recent[::-1], [f"问题{i}" for i in range(1, 6)])
        # 摘要已覆盖被挤出的轮次，窗口放得下剩余部分，不再触发新的摘要。
        self.assertIsNone(chat_sessions.load_summary_batch(self.session_id, 12, 8))

    def test_summary_transcript_drops_citation_markers(self):
        transcript, _ = chat_memory._bounded_transcript(
            ((1, "assistant", "年假为 5 天 [来源 2]，满一年增加【来源 3】。"),)
        )

        self.assertEqual(transcript, "助手：年假为 5 天 ，满一年增加。")

    def test_oversized_summary_is_cut_at_a_complete_line(self):
        summary = "- 第一条内容\n- 第二条内容\n- 第三条被截断的内容"

        truncated = chat_memory._truncate_summary(summary, len(summary) - 3)

        self.assertEqual(truncated, "- 第一条内容\n- 第二条内容")
        self.assertEqual(chat_memory._truncate_summary("- 很短", 100), "- 很短")
