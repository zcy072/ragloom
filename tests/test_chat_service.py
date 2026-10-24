import asyncio
import json
import sqlite3
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

try:
    from langchain_core.documents import Document

    import app_storage
    from services.chat import planner as chat_planner
    from services.chat import query as chat_query
    from services.chat import ranking as chat_ranking
    from services.chat import rewrite as chat_rewrite
    from services.chat import service as chat_service
    from services.chat import sessions as chat_sessions
    from services.core import config as application_config
    from services.core import llm_config as model_config
    from services.core.config import RETRIEVAL_RERANK_MAX_CANDIDATES
    from services.core.container import ApplicationContainer
    from services.core.errors import AccessDeniedError, ResourceNotFoundError
    from services.indexing import chunking
    from services.retrieval import context as context_builder
except ModuleNotFoundError:
    Document = None
    chat_service = None
    chunking = None
    context_builder = None

from tests.support import IsolatedStorageTestCase


@unittest.skipUnless(chat_service and Document, "requires the backend Python dependencies")
class ChatServiceTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        # 建真实 schema 而不是在测试里手写简化表，否则 database.py 的表结构变化不会
        # 让这里失败，测试会对着一份过时的 schema 长期返回假阴性。
        app_storage.initialize_database()
        self.owner = chat_service.SessionOwner(1, "alice")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.executemany(
                "INSERT INTO knowledge_bases (name, created_at, query_rewrite_enabled) "
                "VALUES (?, ?, 0)",
                [
                    ("技术部", application_config.db_now()),
                    ("人力资源部", application_config.db_now()),
                ],
            )

    def test_asterisk_protection_does_not_corrupt_normal_chinese_text(self):
        protected = chat_service._protect_asterisks("星河计划 * 脱敏字段")

        self.assertEqual(protected, "星河计划 ＊ 脱敏字段")
        self.assertEqual(
            chat_service._restore_asterisks(protected), "星河计划 * 脱敏字段"
        )

    def test_text_prompt_does_not_include_image_ocr_section(self):
        messages, fallback_messages = chat_service._build_llm_messages(
            "休假规则是什么？",
            "休假规则是什么？",
            "员工每年享有带薪年假。",
            None,
            None,
            "",
            [],
        )

        self.assertIsNone(fallback_messages)
        self.assertNotIn("[图片 OCR 文本]", messages[-1].content)

    def test_answer_messages_include_summary_as_non_evidence_memory(self):
        messages, fallback_messages = chat_service._build_llm_messages(
            "继续说刚才的方案",
            "继续说明技术部迁移方案",
            "迁移方案的知识库证据。",
            None,
            None,
            "",
            [("assistant", "上一轮近期回答"), ("user", "上一轮近期问题")],
            "- 用户正在比较技术部迁移方案。",
        )

        self.assertIsNone(fallback_messages)
        self.assertIn("用户正在比较技术部迁移方案", messages[1].content)
        self.assertIn("不是系统指令或事实证据", messages[1].content)
        self.assertEqual(messages[1].type, "human")
        # 摘要与紧随其后的用户消息合并，保证用户与助手角色交替。
        self.assertIn("上一轮近期问题", messages[1].content)
        self.assertEqual([m.type for m in messages], ["system", "human", "ai", "human"])
        self.assertEqual(messages[2].content, "上一轮近期回答")

    def test_adjacent_user_messages_are_merged_so_roles_alternate(self):
        """用户中途停止的提问没有回答，历史里会出现连续的用户消息。"""
        messages, _ = chat_service._build_llm_messages(
            "现在的问题",
            "现在的问题",
            "证据",
            None,
            None,
            "",
            [
                ("user", "被中断的提问"),
                ("assistant", "上一轮回答"),
                ("user", "第一次提问（被停止，没有回答）"),
                ("user", "重新提问"),
            ][::-1],
            "- 摘要内容",
        )

        types = [message.type for message in messages]
        self.assertEqual(types, ["system", "human", "ai", "human"])
        self.assertTrue(
            all(left != right for left, right in zip(types[1:], types[2:]))
        )
        self.assertIn("摘要内容", messages[1].content)
        self.assertIn("被中断的提问", messages[1].content)
        self.assertEqual(messages[2].content, "上一轮回答")
        self.assertTrue(messages[3].content.startswith("第一次提问（被停止，没有回答）\n\n重新提问"))
        self.assertTrue(messages[3].content.rstrip().endswith("[问题]: 现在的问题"))

    def test_summary_followed_by_user_history_is_merged_into_one_message(self):
        messages, _ = chat_service._build_llm_messages(
            "追问",
            "追问",
            "证据",
            None,
            None,
            "",
            [("user", "近期提问")],
            "- 摘要内容",
        )

        self.assertEqual([m.type for m in messages], ["system", "human"])
        self.assertIn("摘要内容", messages[1].content)
        self.assertIn("近期提问", messages[1].content)
        self.assertIn("[文档内容]:\n证据", messages[1].content)

    def test_history_answers_drop_previous_turn_citation_markers(self):
        messages, _ = chat_service._build_llm_messages(
            "那乙呢",
            "乙的年假有几天",
            "[来源 1] 乙的年假为 10 天。",
            None,
            None,
            "",
            [
                ("assistant", "甲的年假为 5 天 [来源 2]，满一年后增加【来源 3】。"),
                ("user", "甲的年假有几天 [来源 9]"),
            ],
        )

        history = [message.content for message in messages[1:-1]]
        self.assertEqual(history[0], "甲的年假有几天 [来源 9]")
        self.assertEqual(history[1], "甲的年假为 5 天 ，满一年后增加。")

    def test_session_title_comes_from_the_first_question_only(self):
        session_id = chat_sessions.resolve_session(None, self.owner, "技术部").session_id
        first_question = "请问迁移方案的回滚步骤是什么？" + "补充说明" * 10

        chat_sessions.save_user_message(session_id, first_question)
        chat_sessions.save_user_message(session_id, "第二个问题不应改标题")

        title = chat_sessions.list_owned_sessions(self.owner)[0]["title"]
        self.assertTrue(title.startswith("请问迁移方案的回滚步骤是什么？"))
        self.assertTrue(title.endswith("…"))
        self.assertEqual(len(title), chat_sessions.SESSION_TITLE_MAX_CHARS + 1)
        self.assertEqual(chat_sessions.session_title_from_question("  \n "), "新会话")

    def test_explicit_new_session_does_not_reuse_recent_session_memory(self):
        existing_session_id = chat_sessions.resolve_session(
            None, self.owner, "技术部"
        ).session_id
        reused_session_id = chat_sessions.resolve_session(
            None, self.owner, "技术部"
        ).session_id
        new_session_id = chat_sessions.resolve_session(
            None, self.owner, "技术部", force_new=True
        ).session_id

        self.assertEqual(reused_session_id, existing_session_id)
        self.assertNotEqual(new_session_id, existing_session_id)

    def test_chat_request_rejects_existing_and_new_session_together(self):
        with self.assertRaises(ValueError):
            chat_service.ChatRequest(
                question="测试",
                kb_name="技术部",
                session_id="session-1",
                new_session=True,
            )

    def test_new_session_is_removed_when_pre_stream_processing_fails(self):
        """首次请求在返回 SSE 前失败时不应遗留空会话或孤立用户消息。"""
        request = chat_service.ChatRequest(
            question="测试问题", kb_name="技术部", new_session=True
        )

        with patch.object(
            chat_service,
            "select_model_config",
            return_value={"api_key": "key", "base_url": "url", "model": "model"},
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_service, "_plan_query", new=AsyncMock(side_effect=RuntimeError("检索准备失败"))
        ):
            with self.assertRaisesRegex(RuntimeError, "检索准备失败"):
                asyncio.run(chat_service.chat(request, self.owner))

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0], 0)

    def test_existing_session_does_not_save_half_turn_when_pre_stream_processing_fails(self):
        """已有会话在流式前置阶段失败时也不应提前保存用户问题。"""
        session_id = chat_sessions.resolve_session(
            None, self.owner, "技术部"
        ).session_id
        request = chat_service.ChatRequest(
            question="测试问题", kb_name="技术部", session_id=session_id
        )

        with patch.object(
            chat_service,
            "select_model_config",
            return_value={"api_key": "key", "base_url": "url", "model": "model"},
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_service, "rerank", new=AsyncMock(side_effect=RuntimeError("重排失败"))
        ), patch.object(
            chat_service,
            "retrieve_candidates_with_query_fallback",
            new=AsyncMock(return_value=([object()], "测试问题")),
        ):
            with self.assertRaisesRegex(RuntimeError, "重排失败"):
                asyncio.run(chat_service.chat(request, self.owner))

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM chat_messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM chat_sessions WHERE id = ?", (session_id,)
                ).fetchone()[0],
                1,
            )

    def test_image_prompt_includes_ocr_section_in_primary_and_fallback_messages(self):
        messages, fallback_messages = chat_service._build_llm_messages(
            "图片中的编号是什么？",
            "图片中的编号是什么？",
            "相关文档内容",
            "base64-data",
            "image/png",
            "编号 A-1024",
            [],
        )

        self.assertIn("[图片 OCR 文本]:\n编号 A-1024", messages[-1].content[0]["text"])
        self.assertIsNotNone(fallback_messages)
        self.assertIn("[图片 OCR 文本]:\n编号 A-1024", fallback_messages[-1].content)

    def test_existing_session_must_belong_to_current_user_and_knowledge_base(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO chat_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "session-1", 1, "alice", "技术部",
                    "技术部 会话", "2026-01-01", "2026-01-01",
                ),
            )

        self.assertEqual(
            chat_sessions.resolve_session(
                "session-1", self.owner, "技术部"
            ).session_id,
            "session-1",
        )
        with self.assertRaises(ResourceNotFoundError) as missing:
            chat_sessions.resolve_session("missing", self.owner, "技术部")
        self.assertEqual(missing.exception.code, "chat_session_not_found")
        with self.assertRaises(AccessDeniedError) as forbidden:
            chat_sessions.resolve_session("session-1", self.owner, "人力资源部")
        self.assertEqual(forbidden.exception.code, "chat_session_access_denied")

    def test_same_username_with_a_different_account_id_cannot_reuse_session(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO chat_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "session-identity", 1, "alice", "技术部",
                    "技术部 会话", "2026-01-01", "2026-01-01",
                ),
            )

        other_account = chat_service.SessionOwner(2, "alice")
        with self.assertRaises(AccessDeniedError) as forbidden:
            chat_sessions.resolve_session(
                "session-identity", other_account, "技术部"
            )

        self.assertEqual(forbidden.exception.code, "chat_session_access_denied")

    def test_chat_rejects_a_deleted_knowledge_base(self):
        request = chat_service.ChatRequest(
            question="测试问题", kb_name="已删除知识库"
        )
        with patch.object(
            chat_service,
            "select_model_config",
            return_value={"api_key": "test-key", "base_url": "https://example.test/v1", "model": "test"},
        ):
            with self.assertRaises(ResourceNotFoundError) as raised:
                asyncio.run(chat_service.chat(request, self.owner))
        self.assertEqual(raised.exception.code, "knowledge_base_not_found")

    def test_chat_validation_uses_the_explicit_application_container(self):
        request = chat_service.ChatRequest(question="测试", kb_name="技术部")
        documents = SimpleNamespace(knowledge_base_exists=lambda _name: False)
        settings = SimpleNamespace()
        container = ApplicationContainer(
            documents=documents,
            knowledge_base_settings=settings,
            select_model=lambda _provider: {"api_key": "configured"},
        )

        with self.assertRaises(ResourceNotFoundError) as raised:
            chat_service._validate_and_normalize(request, container)

        self.assertEqual(raised.exception.code, "knowledge_base_not_found")

    def test_session_storage_uses_the_explicit_container_database_path(self):
        connection = MagicMock()
        connection.execute.return_value.fetchall.return_value = []
        context = MagicMock()
        context.__enter__.return_value = connection
        container = ApplicationContainer(
            documents=SimpleNamespace(),
            knowledge_base_settings=SimpleNamespace(),
            database_path=lambda: "/tmp/explicit-chat.sqlite3",
        )

        with patch.object(
            chat_sessions.application_db, "connect", return_value=context
        ) as connect:
            sessions = chat_sessions.list_owned_sessions(
                self.owner, container=container
            )

        self.assertEqual(sessions, [])
        connect.assert_called_once_with("/tmp/explicit-chat.sqlite3")

    def test_session_rows_are_mapped_to_history_contracts(self):
        session_id = chat_sessions.resolve_session(
            None, self.owner, "技术部", force_new=True
        ).session_id
        chat_sessions.save_user_message(session_id, "制度是什么？")
        chat_sessions.save_assistant_message(
            session_id,
            "请参见制度。[来源 1]",
            [{"id": 1, "source": "制度.pdf"}],
        )

        summaries = chat_sessions.list_owned_sessions(self.owner)
        kb_name, messages = chat_sessions.load_owned_session_messages(
            session_id, self.owner
        )

        self.assertEqual(set(summaries[0]), {"id", "username", "kb_name", "title", "date"})
        self.assertEqual(kb_name, "技术部")
        self.assertEqual(messages[0], {"role": "user", "content": "制度是什么？", "sources": []})
        self.assertEqual(messages[1]["sources"], [{"id": 1, "source": "制度.pdf"}])

    def test_chat_accepts_a_configured_selected_provider_when_active_model_has_no_key(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO models_config VALUES (?, ?, ?, ?, ?)",
                ("备用模型", "selected-key", "https://example.test/v1", "selected-model", 0),
            )
        request = chat_service.ChatRequest(
            question="测试问题",
            kb_name="技术部",
            model_provider="备用模型",
        )
        with patch.object(
            model_config,
            "get_active_model_config",
            return_value={"api_key": ""},
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(chat_query, "retrieve_rerank_candidates", new=AsyncMock(return_value=[])), patch.object(
            chat_service, "rerank", new=AsyncMock(return_value=[])
        ):
            stream = asyncio.run(chat_service.chat(request, self.owner))
        self.assertIsNotNone(stream.events)

    def test_retrieval_fallback_keeps_nonempty_rewritten_results(self):
        primary = [Document(page_content="改写命中", metadata={"source": "手册.pdf"})]
        retrieve = AsyncMock(return_value=primary)

        with patch.object(chat_query, "retrieve_rerank_candidates", retrieve):
            documents, used_query = asyncio.run(
                chat_service.retrieve_candidates_with_query_fallback(
                    "技术部", "原问题", "改写后的问题", False
                )
            )

        self.assertEqual(documents, primary)
        self.assertEqual(used_query, "改写后的问题")
        retrieve.assert_awaited_once_with("技术部", "改写后的问题", False, None)

    def test_retrieval_fallback_uses_original_question_after_empty_rewrite(self):
        fallback = [Document(page_content="原问题命中", metadata={"source": "手册.pdf"})]
        retrieve = AsyncMock(side_effect=[[], fallback])

        with patch.object(chat_query, "retrieve_rerank_candidates", retrieve):
            documents, used_query = asyncio.run(
                chat_service.retrieve_candidates_with_query_fallback(
                    "技术部", "原问题", "改写后的问题", False
                )
            )

        self.assertEqual(documents, fallback)
        self.assertEqual(used_query, "原问题")
        self.assertEqual(
            retrieve.await_args_list,
            [
                (("技术部", "改写后的问题", False, None), {}),
                (("技术部", "原问题", False, None), {}),
            ],
        )

    def test_rerank_uses_documents_and_bounds_candidate_count(self):
        documents = [
            Document(page_content=f"候选内容 {index}", metadata={"source": "手册.pdf"})
            for index in range(RETRIEVAL_RERANK_MAX_CANDIDATES + 5)
        ]

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "results": [
                        {"index": RETRIEVAL_RERANK_MAX_CANDIDATES - 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.1},
                    ]
                }

        client = SimpleNamespace(post=AsyncMock(return_value=FakeResponse()))
        with patch.object(
            chat_ranking, "_get_rerank_client", new=AsyncMock(return_value=client)
        ), patch.object(
            model_config,
            "get_siliconflow_api_key",
            return_value="test-key",
        ):
            ranked = asyncio.run(chat_service.rerank("候选内容", documents))

        payload = client.post.await_args.kwargs["json"]
        self.assertIn("documents", payload)
        self.assertNotIn("texts", payload)
        self.assertEqual(len(payload["documents"]), RETRIEVAL_RERANK_MAX_CANDIDATES)
        self.assertEqual(payload["top_n"], chat_ranking.RERANK_TOP_N)
        self.assertEqual(
            ranked[0].page_content,
            f"候选内容 {RETRIEVAL_RERANK_MAX_CANDIDATES - 1}",
        )

    def test_rerank_empty_response_keeps_bounded_candidates(self):
        self._assert_empty_rerank_fallback()

    def test_identifier_ties_preserve_scores_and_reject_invalid_results(self):
        diagnostics = chat_ranking.RetrievalDiagnostics("AB1234 2026 manual")
        documents = [Document(page_content=text, metadata={"source": "test.pdf"})
                     for text in ["AB12345", "ab1234", "2026 manual", "unrelated"]]
        response = SimpleNamespace(status_code=200, json=lambda: {"results": [
            {"index": 0, "relevance_score": 0.8},
            {"index": 1, "relevance_score": 0.8},
            {"index": 1, "relevance_score": 0.9},
            {"index": True, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.85},
            {"index": 3, "relevance_score": float("nan")},
        ]})
        client = SimpleNamespace(post=AsyncMock(return_value=response))
        with patch.object(chat_ranking, "_get_rerank_client", new=AsyncMock(return_value=client)), patch.object(
            model_config, "get_siliconflow_api_key", return_value="test-key"
        ):
            ranked = asyncio.run(chat_service.rerank("AB1234 2026 manual", documents, diagnostics=diagnostics))
        self.assertEqual([doc.page_content for doc in ranked], ["2026 manual", "ab1234", "AB12345"])
        self.assertEqual(ranked[0].metadata["exact_identifiers"], [])
        self.assertEqual(ranked[1].metadata["relevance_score"], 0.8)
        self.assertEqual(ranked[1].metadata["exact_identifiers"], ["AB1234"])
        self.assertNotIn("relevance_score", documents[1].metadata)
        self.assertEqual(diagnostics.rerank_status, "success")
        self.assertEqual(diagnostics.ranking[1]["model_score"], 0.8)
        self.assertEqual(diagnostics.ranking[1]["exact_identifiers"], ["AB1234"])

    def test_rerank_service_failure_preserves_candidates_and_reports_degradation(self):
        documents = [Document(page_content="AB1234", metadata={"source": "manual.pdf"})]
        diagnostics = chat_ranking.RetrievalDiagnostics("AB1234")
        client = SimpleNamespace(post=AsyncMock(side_effect=RuntimeError("offline")))
        with patch.object(chat_ranking, "_get_rerank_client", new=AsyncMock(return_value=client)), patch.object(
            model_config, "get_siliconflow_api_key", return_value="test-key"
        ):
            ranked = asyncio.run(chat_service.rerank("AB1234", documents, diagnostics=diagnostics))
        self.assertEqual(ranked, documents)
        self.assertEqual(diagnostics.rerank_status, "degraded")
        self.assertEqual(diagnostics.ranking, [])

    def _assert_empty_rerank_fallback(self):
        documents = [
            Document(page_content=f"候选内容 {index}", metadata={"source": "手册.pdf"})
            for index in range(RETRIEVAL_RERANK_MAX_CANDIDATES + 2)
        ]

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"results": []}

        client = SimpleNamespace(post=AsyncMock(return_value=FakeResponse()))
        with patch.object(
            chat_ranking, "_get_rerank_client", new=AsyncMock(return_value=client)
        ), patch.object(
            model_config,
            "get_siliconflow_api_key",
            return_value="test-key",
        ):
            ranked = asyncio.run(chat_service.rerank("测试", documents))

        self.assertEqual(ranked, documents[: chat_ranking.RERANK_TOP_N])

    def test_rerank_failure_returns_same_count_as_success(self):
        documents = [
            Document(page_content=f"候选内容 {index}", metadata={"source": f"文件{index}.pdf"})
            for index in range(RETRIEVAL_RERANK_MAX_CANDIDATES)
        ]
        client = SimpleNamespace(post=AsyncMock(side_effect=RuntimeError("offline")))
        with patch.object(
            chat_ranking, "_get_rerank_client", new=AsyncMock(return_value=client)
        ), patch.object(
            model_config, "get_siliconflow_api_key", return_value="test-key"
        ), self.assertLogs(chat_ranking.logger, level="WARNING"):
            ranked = asyncio.run(chat_service.rerank("测试", documents))

        self.assertEqual(ranked, documents[: chat_ranking.RERANK_TOP_N])

    def test_rerank_candidate_selection_prioritizes_document_route_diversity(self):
        diabetes_fixed = [
            Document(
                page_content=f"糖尿病固定块 {index}",
                metadata={"source": "糖尿病指南.pdf", "chunk_route": chunking.CHUNK_ROUTE_FIXED},
            )
            for index in range(12)
        ]
        diagnosis = Document(
            page_content="高血压诊断标准参见指南",
            metadata={
                "source": "高血压标准.pdf",
                "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED,
            },
        )
        documents = diabetes_fixed + [
            Document(
                page_content=f"高血压结构块 {index}",
                metadata={
                    "source": "高血压标准.pdf",
                    "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED,
                },
            )
            for index in range(5)
        ] + [diagnosis]

        selected = chat_ranking.select_rerank_candidates(documents)

        self.assertIn(diagnosis, selected)
        self.assertEqual(len(selected), len(documents))
        self.assertEqual(
            sum(document.metadata["source"] == "糖尿病指南.pdf" for document in selected[:12]),
            chat_ranking.RERANK_MAX_PER_SOURCE_ROUTE,
        )

    def test_rerank_candidate_selection_prioritizes_distinct_parents(self):
        repeated_parent = [
            Document(
                page_content=f"同一父块的子块 {index}",
                metadata={
                    "source": "员工手册.pdf",
                    "chunk_route": chunking.CHUNK_ROUTE_FIXED,
                    "parent_id": "parent-1",
                },
            )
            for index in range(8)
        ]
        alternatives = [
            Document(
                page_content=f"独立父块 {index}",
                metadata={
                    "source": "员工手册.pdf",
                    "chunk_route": chunking.CHUNK_ROUTE_FIXED,
                    "parent_id": f"parent-{index + 2}",
                },
            )
            for index in range(2)
        ]

        selected = chat_ranking.select_rerank_candidates(repeated_parent + alternatives)
        first_parent_ids = [document.metadata["parent_id"] for document in selected[:4]]

        self.assertEqual(first_parent_ids, ["parent-1", "parent-2", "parent-3", "parent-1"])
        self.assertLessEqual(
            first_parent_ids.count("parent-1"), chat_ranking.RERANK_MAX_PER_PARENT
        )

    def test_retrieval_supplements_vector_results_with_lexical_candidates(self):
        lexical = Document(
            page_content="高血压诊断标准参见指南",
            metadata={"source": "标准.pdf", "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED},
        )
        semantic = Document(
            page_content="语义候选内容",
            metadata={"source": "指南.pdf", "chunk_route": chunking.CHUNK_ROUTE_FIXED},
        )
        cached = SimpleNamespace(invoke=lambda _question: [semantic])

        with patch.object(
            chat_query.get_application_container().documents,
            "has_filters",
            return_value=False,
        ), patch.object(
            chat_query.retrieval_service, "request_source_snapshot", return_value=nullcontext()
        ), patch.object(
            chat_query.retrieval_service, "get_cached_retriever", return_value=cached
        ), patch.object(
            chat_query.retrieval_service,
            "retrieve_lexical_candidates",
            return_value=[lexical],
        ), patch.object(
            chat_query.retrieval_service,
            "deduplicate_chunk_routes",
            side_effect=lambda docs: docs,
        ):
            documents = asyncio.run(
                chat_query.retrieve_rerank_candidates("技术部", "高血压诊断标准", False)
            )

        self.assertEqual(documents, [lexical, semantic])

    def test_lexical_candidates_do_not_crowd_semantic_ones_out_of_the_cap(self):
        lexical = [
            Document(
                page_content=f"精确词候选 {index}",
                metadata={"source": f"词法{index}.pdf", "chunk_route": chunking.CHUNK_ROUTE_FIXED},
            )
            for index in range(RETRIEVAL_RERANK_MAX_CANDIDATES)
        ]
        semantic = [
            Document(
                page_content=f"语义候选 {index}",
                metadata={"source": f"语义{index}.pdf", "chunk_route": chunking.CHUNK_ROUTE_FIXED},
            )
            for index in range(5)
        ]
        cached = SimpleNamespace(invoke=lambda _question: semantic)

        with patch.object(
            chat_query.get_application_container().documents,
            "has_filters",
            return_value=False,
        ), patch.object(
            chat_query.retrieval_service, "request_source_snapshot", return_value=nullcontext()
        ), patch.object(
            chat_query.retrieval_service, "get_cached_retriever", return_value=cached
        ), patch.object(
            chat_query.retrieval_service,
            "retrieve_lexical_candidates",
            return_value=lexical,
        ), patch.object(
            chat_query.retrieval_service,
            "deduplicate_chunk_routes",
            side_effect=lambda docs: docs,
        ):
            documents = asyncio.run(
                chat_query.retrieve_rerank_candidates("技术部", "高血压诊断标准", False)
            )

        self.assertEqual(len(documents), RETRIEVAL_RERANK_MAX_CANDIDATES)
        self.assertEqual(documents[0], lexical[0])
        self.assertEqual(documents[1], semantic[0])
        for document in semantic:
            self.assertIn(document, documents)

    def test_source_catalog_keeps_prompt_ids_and_uses_a_query_focused_preview(self):
        spans = [
            context_builder.ContextSpan(
                source="first.pdf",
                content=(
                    "无关开头。" * 40
                    + "高血压诊断标准参见《国家基层高血压防治管理指南》。后续内容。"
                ),
                pages=("0",),
                page_unit="页",
                rank=0,
            ),
            context_builder.ContextSpan(
                source="second.pdf",
                content="第二段上下文",
                pages=("1",),
                page_unit="页",
                rank=1,
            ),
        ]
        sources = chat_service.build_source_catalog(spans, "高血压诊断标准参见什么")
        self.assertEqual(sources[0]["source"], "first.pdf")
        self.assertEqual(sources[0]["page"], "第 1 页")
        self.assertEqual(sources[0]["id"], 1)
        self.assertEqual(sources[0]["citation_ids"], [1])
        self.assertIn("高血压诊断标准参见", sources[0]["content"])
        self.assertTrue(sources[0]["content"].startswith("..."))
        self.assertEqual(sources[1]["id"], 2)

    def test_only_cited_sources_are_returned_and_same_document_is_grouped(self):
        catalog = [
            {
                "id": 1,
                "citation_ids": [1],
                "source": "手册.pdf",
                "page": "第 4 页",
                "context_title": "诊断",
                "content": "诊断依据",
            },
            {
                "id": 2,
                "citation_ids": [2],
                "source": "手册.pdf",
                "page": "第 8 页",
                "context_title": "参考文献",
                "content": "参考依据",
            },
            {
                "id": 3,
                "citation_ids": [3],
                "source": "无关.pdf",
                "page": "第 1 页",
                "context_title": "无关",
                "content": "无关内容",
            },
        ]

        sources = chat_service.select_cited_sources("结论。[来源 1] 补充。[来源 2]", catalog)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source"], "手册.pdf")
        self.assertEqual(sources[0]["citation_ids"], [1, 2])
        self.assertEqual(sources[0]["page"], "第 4 页、第 8 页")
        self.assertEqual(chat_service.select_cited_sources("没有引用标记", catalog), [])

    def test_grouped_citations_keep_independent_evidence(self):
        spans = [context_builder.ContextSpan(
            source="manual.pdf", content=text, pages=(page,), page_unit="页",
            rank=index, kb_name="研发部",
        ) for index, (text, page) in enumerate([("第一证据", "0"), ("第二证据", "7")])]
        catalog = chat_service.build_source_catalog(spans)
        sources = chat_service.select_cited_sources("[来源 2] [来源 1] [来源 99]", catalog)
        self.assertEqual(len(sources), 1)
        evidence = sources[0]["evidence"]
        self.assertEqual([item["id"] for item in evidence], [2, 1])
        self.assertEqual([item["page_numbers"] for item in evidence], [[8], [1]])
        self.assertEqual([item["content"] for item in evidence], ["第二证据", "第一证据"])

    def test_same_filename_in_different_knowledge_bases_is_not_grouped(self):
        spans = [
            context_builder.ContextSpan(
                source="制度.pdf",
                content="部门制度",
                pages=("0",),
                page_unit="页",
                rank=0,
                kb_name="研发部",
            ),
            context_builder.ContextSpan(
                source="制度.pdf",
                content="通用制度",
                pages=("1",),
                page_unit="页",
                rank=1,
                kb_name="通用知识库",
            ),
        ]
        catalog = chat_service.build_source_catalog(spans, "制度")

        sources = chat_service.select_cited_sources("部门依据。[来源 1] 通用依据。[来源 2]", catalog)

        self.assertEqual(len(sources), 2)
        self.assertEqual([source["kb_name"] for source in sources], ["研发部", "通用知识库"])

    def test_chat_stream_and_history_keep_only_answer_citations(self):
        class FakeStreamingModel:
            async def astream(self, _messages):
                yield SimpleNamespace(content="结论。[来源 2]")

        spans = [
            context_builder.ContextSpan(
                source="手册.pdf",
                content="第一条上下文",
                pages=("0",),
                page_unit="页",
                rank=0,
            ),
            context_builder.ContextSpan(
                source="手册.pdf",
                content="第二条上下文，包含结论依据。",
                pages=("3",),
                page_unit="页",
                rank=1,
            ),
        ]
        request = chat_service.ChatRequest(question="结论是什么", kb_name="技术部")

        async def collect_events(stream):
            chunks = []
            async for chunk in stream.events:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
            return chunks

        with patch.object(
            chat_service, "select_model_config", return_value={"api_key": "test-key", "base_url": "https://example.test/v1", "model": "test"}
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_query, "retrieve_rerank_candidates", new=AsyncMock(return_value=[])
        ), patch.object(chat_service, "rerank", new=AsyncMock(return_value=[])), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=spans
        ), patch.object(chat_service, "create_answer_llm", return_value=FakeStreamingModel()):
            stream = asyncio.run(chat_service.chat(request, self.owner))
            events = asyncio.run(collect_events(stream))

        end_event = json.loads(next(event[6:] for event in events if '"is_end": true' in event))
        self.assertEqual(end_event["sources"], [
            {
                "id": 2,
                "citation_ids": [2],
                "source": "手册.pdf",
                "page": "第 4 页",
                "context_title": "",
                "content": "第二条上下文，包含结论依据。",
                "evidence": [{
                    "id": 2, "source": "手册.pdf", "kb_name": "",
                    "page": "第 4 页", "page_numbers": [4],
                    "context_title": "", "content": "第二条上下文，包含结论依据。",
                }],
            }
        ])
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            sources = json.loads(
                conn.execute(
                    "SELECT sources FROM chat_messages WHERE role = 'assistant' ORDER BY id DESC LIMIT 1"
                ).fetchone()[0]
            )
        self.assertEqual(sources, end_event["sources"])

    def test_chat_cancellation_before_first_token_does_not_save_an_empty_answer(self):
        class CancelledStreamingModel:
            async def astream(self, _messages):
                raise asyncio.CancelledError()
                yield SimpleNamespace(content="unreachable")

        spans = [
            context_builder.ContextSpan(
                source="手册.pdf",
                content="用于触发回答模型的上下文。",
                pages=("0",),
                page_unit="页",
                rank=0,
            )
        ]
        request = chat_service.ChatRequest(question="这个问题会被取消", kb_name="技术部")

        async def consume(stream):
            async for _ in stream.events:
                pass

        with patch.object(
            chat_service,
            "select_model_config",
            return_value={
                "api_key": "test-key",
                "base_url": "https://example.test/v1",
                "model": "test",
            },
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_query, "retrieve_rerank_candidates", new=AsyncMock(return_value=[])
        ), patch.object(
            chat_service, "rerank", new=AsyncMock(return_value=[])
        ), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=spans
        ), patch.object(
            chat_service, "create_answer_llm", return_value=CancelledStreamingModel()
        ):
            stream = asyncio.run(chat_service.chat(request, self.owner))
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(consume(stream))

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            roles_and_content = conn.execute(
                "SELECT role, content FROM chat_messages ORDER BY id"
            ).fetchall()
        self.assertEqual(roles_and_content, [("user", "这个问题会被取消")])

    def test_multimodal_partial_stream_is_not_restarted_with_text_fallback(self):
        class PartialFailureStreamingModel:
            def __init__(self):
                self.calls = 0

            async def astream(self, _messages):
                self.calls += 1
                if self.calls == 1:
                    yield SimpleNamespace(content="已经输出的部分回答")
                    raise RuntimeError("视觉连接中断")
                yield SimpleNamespace(content="不应再次输出的降级回答")

        model = PartialFailureStreamingModel()
        source_catalog = [
            {
                "id": 1,
                "citation_ids": [1],
                "source": "手册.pdf",
                "page": "第 1 页",
                "context_title": "",
                "content": "回答依据",
            }
        ]

        async def collect_events():
            return [
                event
                async for event in chat_service._stream_answer(
                    "session-partial",
                    {
                        "api_key": "test-key",
                        "base_url": "https://example.test/v1",
                        "model": "test",
                    },
                    source_catalog,
                    [SimpleNamespace(content="multimodal")],
                    [SimpleNamespace(content="text-only")],
                    "base64-image",
                )
            ]

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO chat_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "session-partial",
                    1,
                    "alice",
                    "技术部",
                    "测试",
                    "2026-01-01",
                    "2026-01-01",
                ),
            )

        with patch.object(chat_service, "create_answer_llm", return_value=model):
            events = asyncio.run(collect_events())

        self.assertEqual(model.calls, 1)
        chunks = [
            json.loads(event[6:])["chunk"]
            for event in events
            if '"chunk"' in event
        ]
        self.assertEqual(chunks, ["已经输出的部分回答"])
        terminal = json.loads(next(event[6:] for event in events if '"is_end": true' in event))
        self.assertEqual(terminal["final_answer"], "已经输出的部分回答")
        self.assertTrue(terminal["interrupted"])
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            stored = conn.execute(
                "SELECT content FROM chat_messages WHERE session_id = ? AND role = 'assistant'",
                ("session-partial",),
            ).fetchone()[0]
        self.assertEqual(stored, terminal["final_answer"])

    def test_generation_failure_emits_an_error_event_and_saves_no_assistant_message(self):
        class FailingStreamingModel:
            async def astream(self, _messages):
                raise RuntimeError("https://upstream.test/v1 secret-detail")
                yield SimpleNamespace(content="unreachable")

        source_catalog = [
            {
                "id": 1,
                "citation_ids": [1],
                "source": "手册.pdf",
                "page": "第 1 页",
                "context_title": "",
                "content": "回答依据",
            }
        ]
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO chat_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("session-failed", 1, "alice", "技术部", "测试", "2026-01-01", "2026-01-01"),
            )
        # 一条更早的完整问答，证明撤回只影响本轮没有回答的问题。
        chat_sessions.save_user_message("session-failed", "更早的问题")
        chat_sessions.save_assistant_message("session-failed", "更早的回答", [])
        user_message_id = chat_sessions.save_user_message("session-failed", "本轮问题")

        async def collect_events():
            return [
                event
                async for event in chat_service._stream_answer(
                    "session-failed",
                    {"api_key": "test-key", "base_url": "https://example.test/v1", "model": "test"},
                    source_catalog,
                    [SimpleNamespace(content="multimodal")],
                    [SimpleNamespace(content="text-only")],
                    "base64-image",
                    user_message_id=user_message_id,
                )
            ]

        with patch.object(chat_service, "create_answer_llm", return_value=FailingStreamingModel()):
            events = asyncio.run(collect_events())

        self.assertEqual(len(events), 1)
        payload = json.loads(events[0][6:])
        self.assertEqual(payload["error"], chat_service.ANSWER_ERROR_MESSAGE)
        self.assertTrue(payload["is_end"])
        self.assertEqual(payload["session_id"], "session-failed")
        self.assertNotIn("secret-detail", events[0])
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            saved = conn.execute(
                "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY id",
                ("session-failed",),
            ).fetchall()
        self.assertEqual(saved, [("user", "更早的问题"), ("assistant", "更早的回答")])

    def test_generation_failure_in_a_new_session_removes_the_empty_session(self):
        class FailingStreamingModel:
            async def astream(self, _messages):
                raise RuntimeError("upstream down")
                yield SimpleNamespace(content="unreachable")

        resolution = chat_sessions.resolve_session(None, self.owner, "技术部")
        user_message_id = chat_sessions.save_user_message(resolution.session_id, "第一个问题")

        async def collect_events():
            return [
                event
                async for event in chat_service._stream_answer(
                    resolution.session_id,
                    {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
                    [{"id": 1, "citation_ids": [1], "source": "手册.pdf", "page": "", "context_title": "", "content": "依据"}],
                    [SimpleNamespace(content="q")],
                    None,
                    None,
                    user_message_id=user_message_id,
                )
            ]

        with patch.object(chat_service, "create_answer_llm", return_value=FailingStreamingModel()):
            events = asyncio.run(collect_events())

        payload = json.loads(events[0][6:])
        self.assertEqual(payload["error"], chat_service.ANSWER_ERROR_MESSAGE)
        # 会话已被删除，不能再把它的 ID 交给前端当作当前会话。
        self.assertNotIn("session_id", payload)
        self.assertEqual(chat_sessions.list_owned_sessions(self.owner), [])

    def test_answer_client_creation_failure_discards_the_question(self):
        resolution = chat_sessions.resolve_session(None, self.owner, "技术部")
        user_message_id = chat_sessions.save_user_message(resolution.session_id, "问题")

        async def collect_events():
            return [
                event
                async for event in chat_service._stream_answer(
                    resolution.session_id,
                    {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
                    [],
                    [SimpleNamespace(content="q")],
                    None,
                    None,
                    allow_answer_without_evidence=True,
                    user_message_id=user_message_id,
                )
            ]

        with patch.object(
            chat_service, "create_answer_llm", side_effect=ValueError("bad config")
        ):
            events = asyncio.run(collect_events())

        payload = json.loads(events[0][6:])
        self.assertEqual(payload["error"], chat_service.ANSWER_ERROR_MESSAGE)
        self.assertNotIn("bad config", events[0])
        self.assertNotIn("session_id", payload)
        self.assertEqual(chat_sessions.list_owned_sessions(self.owner), [])

    def test_partial_answer_keeps_the_user_message(self):
        class BreaksAfterFirstChunk:
            async def astream(self, _messages):
                yield SimpleNamespace(content="部分")
                raise RuntimeError("connection reset")

        resolution = chat_sessions.resolve_session(None, self.owner, "技术部")
        user_message_id = chat_sessions.save_user_message(resolution.session_id, "问题")

        async def collect_events():
            return [
                event
                async for event in chat_service._stream_answer(
                    resolution.session_id,
                    {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
                    [],
                    [SimpleNamespace(content="q")],
                    None,
                    None,
                    allow_answer_without_evidence=True,
                    user_message_id=user_message_id,
                )
            ]

        with patch.object(chat_service, "create_answer_llm", return_value=BreaksAfterFirstChunk()):
            asyncio.run(collect_events())

        _, messages = chat_sessions.load_owned_session_messages(
            resolution.session_id, self.owner
        )
        self.assertEqual([(m["role"], m["content"]) for m in messages], [("user", "问题"), ("assistant", "部分")])

    def test_discard_user_message_only_removes_that_user_message(self):
        session_id = chat_sessions.resolve_session(None, self.owner, "技术部").session_id
        chat_sessions.save_user_message(session_id, "问题一")
        chat_sessions.save_assistant_message(session_id, "回答一", [])
        second_id = chat_sessions.save_user_message(session_id, "问题二")

        # 还有其他消息时只删这一条，会话保留。
        self.assertFalse(chat_sessions.discard_user_message(session_id, second_id))
        # 不存在的消息 ID 是空操作。
        self.assertFalse(chat_sessions.discard_user_message(session_id, 10_000))
        _, messages = chat_sessions.load_owned_session_messages(session_id, self.owner)
        self.assertEqual([m["content"] for m in messages], ["问题一", "回答一"])

        # 会话里最后一条消息被撤回后，空会话连同摘要记忆一起删除。
        lone_session_id = chat_sessions.resolve_session(
            None, self.owner, "人力资源部", force_new=True
        ).session_id
        lone_id = chat_sessions.save_user_message(lone_session_id, "孤立问题")
        self.assertTrue(chat_sessions.discard_user_message(lone_session_id, lone_id))
        self.assertEqual(
            [s["id"] for s in chat_sessions.list_owned_sessions(self.owner)], [session_id]
        )

    def test_chat_returns_no_answer_without_retrieval_evidence(self):
        request = chat_service.ChatRequest(question="没有依据的问题", kb_name="技术部")

        async def collect_events(stream):
            chunks = []
            async for chunk in stream.events:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
            return chunks

        with patch.object(
            chat_service,
            "select_model_config",
            return_value={"api_key": "test-key", "base_url": "https://example.test/v1", "model": "test"},
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_query, "retrieve_rerank_candidates", new=AsyncMock(return_value=[])
        ), patch.object(chat_service, "rerank", new=AsyncMock(return_value=[])), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=[]
        ), patch.object(
            chat_service.chat_memory,
            "refresh_conversation_summary",
            new=AsyncMock(return_value=False),
        ) as memory_refresh, patch.object(chat_service, "create_answer_llm") as llm_factory:
            stream = asyncio.run(chat_service.chat(request, self.owner))
            events = asyncio.run(collect_events(stream))
            asyncio.run(stream.refresh_summary())

        llm_factory.assert_not_called()
        memory_refresh.assert_awaited_once()
        end_event = json.loads(next(event[6:] for event in events if '"is_end": true' in event))
        self.assertEqual(end_event["final_answer"], chat_service.NO_EVIDENCE_ANSWER)
        self.assertEqual(end_event["sources"], [])

    def test_chat_passes_a_resolved_follow_up_to_the_answer_model(self):
        class CapturingStreamingModel:
            def __init__(self):
                self.messages = []

            async def astream(self, messages):
                self.messages = messages
                yield SimpleNamespace(content="实习生年假按制度执行。[来源 1]")

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO chat_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "session-follow-up", 1, "alice", "人力资源部",
                    "测试", "2026-01-01", "2026-01-01",
                ),
            )
            conn.executemany(
                "INSERT INTO chat_messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                [
                    ("session-follow-up", "user", "正式员工年假有几天？", "2026-01-01"),
                    ("session-follow-up", "assistant", "正式员工年假按工龄确定。", "2026-01-01"),
                ],
            )
        request = chat_service.ChatRequest(
            question="实习生的呢",
            kb_name="人力资源部",
            session_id="session-follow-up",
        )
        plan = chat_rewrite.RetrievalQueryPlan(
            query="实习生年假有几天",
            used_history=True,
            inherited_focus="年假天数",
        )
        document = Document(page_content="实习生年假规定", metadata={"source": "员工手册.pdf"})
        spans = [
            context_builder.ContextSpan(
                source="员工手册.pdf",
                content="实习生年假规定",
                pages=("0",),
                page_unit="页",
                rank=0,
            )
        ]
        model = CapturingStreamingModel()

        async def collect_events(stream):
            async for _ in stream.events:
                pass

        with patch.object(
            chat_service,
            "select_model_config",
            return_value={"api_key": "test-key", "base_url": "https://example.test/v1", "model": "test"},
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_service.get_application_container().knowledge_base_settings,
            "query_rewrite_enabled",
            return_value=True,
        ), patch.object(
            chat_service, "build_retrieval_query_plan", new=AsyncMock(return_value=plan)
        ), patch.object(
            chat_service,
            "retrieve_candidates_with_query_fallback",
            new=AsyncMock(return_value=([document], plan.query)),
        ), patch.object(
            chat_service, "rerank", new=AsyncMock(return_value=[document])
        ), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=spans
        ), patch.object(chat_service, "create_answer_llm", return_value=model):
            stream = asyncio.run(chat_service.chat(request, self.owner))
            asyncio.run(collect_events(stream))

        final_prompt = model.messages[-1].content
        self.assertIn("[用户当前原话]: 实习生的呢", final_prompt)
        self.assertIn("[系统解析的独立问题]: 实习生年假有几天", final_prompt)

    def test_follow_up_query_rewrite_uses_structured_response(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = SimpleNamespace(
            content=(
                '{"query":"员工请假申请和审批流程","used_history":true,'
                '"inherited_focus":"请假申请和审批流程"}'
            )
        )
        with patch.object(chat_rewrite, "create_rewrite_llm", return_value=llm):
            plan = asyncio.run(
                chat_service.build_retrieval_query_plan(
                    "那它的流程呢",
                    [("user", "员工如何请假？"), ("assistant", "请参考手册。")],
                    {"api_key": "test-key", "base_url": "https://example.test/v1", "model": "test"},
                )
            )
        self.assertEqual(plan.query, "员工请假申请和审批流程")
        self.assertTrue(plan.used_history)
        self.assertEqual(plan.inherited_focus, "请假申请和审批流程")

    def test_query_rewrite_can_resolve_a_follow_up_from_early_summary(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = SimpleNamespace(
            content=(
                '{"query":"乙型号的保修期多久","used_history":true,'
                '"inherited_focus":"保修期"}'
            )
        )
        with patch.object(chat_rewrite, "create_rewrite_llm", return_value=llm):
            plan = asyncio.run(
                chat_service.build_retrieval_query_plan(
                    "乙型号呢？",
                    [],
                    {
                        "api_key": "test-key",
                        "base_url": "https://example.test/v1",
                        "model": "test",
                    },
                    "此前用户询问甲型号的保修期。",
                )
            )

        prompt = llm.ainvoke.await_args.args[0]
        self.assertIn("此前用户询问甲型号的保修期", prompt[1].content)
        self.assertEqual(plan.query, "乙型号的保修期多久")

    def test_query_rewrite_rejects_history_use_without_an_explicit_focus(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = SimpleNamespace(
            content='{"query":"员工请假申请和审批流程","used_history":true}'
        )
        with patch.object(chat_rewrite, "create_rewrite_llm", return_value=llm):
            plan = asyncio.run(
                chat_service.build_retrieval_query_plan(
                    "那它的流程呢",
                    [("user", "员工如何请假？")],
                    {"api_key": "test-key", "base_url": "https://example.test/v1", "model": "test"},
                )
            )
        self.assertEqual(plan, chat_rewrite.original_query_plan("那它的流程呢"))

    def test_query_rewrite_keeps_original_when_history_was_not_used(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = SimpleNamespace(
            content=(
                '{"query":"模型擅自改写的问题","used_history":false,'
                '"inherited_focus":""}'
            )
        )
        with patch.object(chat_rewrite, "create_rewrite_llm", return_value=llm):
            plan = asyncio.run(
                chat_service.build_retrieval_query_plan(
                    "完整问题",
                    [("user", "上一轮问题")],
                    {"api_key": "test-key", "base_url": "https://example.test/v1", "model": "test"},
                )
            )
        self.assertEqual(plan, chat_rewrite.original_query_plan("完整问题"))

    def test_query_rewrite_falls_back_when_response_is_not_json(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = SimpleNamespace(content="这是一段不合规的回答")
        with patch.object(chat_rewrite, "create_rewrite_llm", return_value=llm):
            plan = asyncio.run(
                chat_service.build_retrieval_query_plan(
                    "那它的流程呢",
                    [("user", "员工如何请假？")],
                    {"api_key": "test-key", "base_url": "https://example.test/v1", "model": "test"},
                )
            )
        self.assertEqual(plan, chat_rewrite.original_query_plan("那它的流程呢"))

    def test_query_rewrite_uses_only_recent_history_and_strips_citations(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = SimpleNamespace(
            content=(
                '{"query":"乙型号的保修期多久","used_history":true,'
                '"inherited_focus":"保修期"}'
            )
        )
        history = [
            ("user", "很早以前的问题"),
            ("assistant", "很早以前的回答"),
            ("user", "无关的中间问题"),
            ("assistant", "无关的中间回答"),
            ("user", "甲型号的保修期多久？"),
            ("assistant", "甲型号保修两年。[来源 1]"),
        ]

        with patch.object(chat_rewrite, "create_rewrite_llm", return_value=llm):
            plan = asyncio.run(
                chat_service.build_retrieval_query_plan(
                    "乙型号呢？",
                    history,
                    {"api_key": "test-key", "base_url": "https://example.test/v1", "model": "test"},
                )
            )

        prompt = llm.ainvoke.await_args.args[0]
        self.assertNotIn("很早以前", prompt[1].content)
        self.assertIn("甲型号的保修期多久", prompt[1].content)
        self.assertNotIn("[来源 1]", prompt[1].content)
        self.assertEqual(plan.query, "乙型号的保修期多久")
        self.assertTrue(plan.used_history)
        self.assertEqual(plan.inherited_focus, "保修期")

    def test_heartbeat_wrapper_pings_during_silence_without_cancelling_upstream(self):
        async def slow_events():
            yield "data: a\n\n"
            await asyncio.sleep(0.12)
            yield "data: b\n\n"

        async def collect():
            return [item async for item in chat_service.with_heartbeat(slow_events(), 0.03)]

        items = asyncio.run(collect())
        self.assertEqual([item for item in items if item != chat_service.SSE_HEARTBEAT],
                         ["data: a\n\n", "data: b\n\n"])
        self.assertGreaterEqual(items.count(chat_service.SSE_HEARTBEAT), 2)

    def test_heartbeat_wrapper_closes_upstream_when_client_disconnects(self):
        closed = []

        async def events():
            try:
                yield "data: a\n\n"
                await asyncio.sleep(10)
                yield "data: never\n\n"
            finally:
                closed.append(True)

        async def consume_one_then_close():
            wrapped = chat_service.with_heartbeat(events(), 0.03)
            first = await wrapped.__anext__()
            await wrapped.aclose()
            return first

        self.assertEqual(asyncio.run(consume_one_then_close()), "data: a\n\n")
        self.assertEqual(closed, [True])

    def test_complete_questions_with_demonstratives_do_not_trigger_rewrite(self):
        history = [("user", "员工如何请假？"), ("assistant", "请参考手册。")]
        for question in (
            "该制度适用于哪些部门的员工以及外包人员是否也在范围内",
            "这份合同约定的违约金比例是多少以及如何计算滞纳金",
            "在试用期内离职的员工是否可以领取当月全额工资",
        ):
            with self.subTest(question=question):
                self.assertFalse(chat_rewrite._should_rewrite_query(question, history))

    def test_follow_up_questions_still_trigger_rewrite(self):
        history = [("user", "员工如何请假？")]
        for question in ("那它的流程呢", "乙型号呢？", "上面提到的审批需要几天", "还有别的条件吗"):
            with self.subTest(question=question):
                self.assertTrue(chat_rewrite._should_rewrite_query(question, history))
        self.assertFalse(chat_rewrite._should_rewrite_query("那它的流程呢", []))
        self.assertTrue(chat_rewrite._should_rewrite_query("乙型号呢", [], "此前询问甲型号。"))

    # ------------------------------------------------------------------
    # 书面化补充查询
    # ------------------------------------------------------------------

    _MODEL = {"api_key": "test-key", "base_url": "https://example.test/v1", "model": "test"}

    def _variant_for(self, query, content):
        llm = AsyncMock()
        llm.ainvoke.return_value = SimpleNamespace(content=content)
        with patch.object(chat_rewrite, "create_rewrite_llm", return_value=llm):
            return asyncio.run(chat_rewrite.build_query_variant(query, self._MODEL))

    def test_query_variant_accepts_a_written_style_rewrite(self):
        variant = self._variant_for("请假要啥手续", '{"query":"员工请假的申请与审批流程"}')
        self.assertEqual(variant, "员工请假的申请与审批流程")

    def test_query_variant_drops_identical_or_unusable_results(self):
        self.assertEqual(self._variant_for("年假几天", '{"query":"年假几天"}'), "")
        self.assertEqual(self._variant_for("年假几天", "不是 JSON"), "")
        self.assertEqual(self._variant_for("年假几天", '{"query":""}'), "")
        self.assertEqual(self._variant_for("年假几天", '{"query":123}'), "")

    def test_query_variant_rejects_results_that_lose_identifiers(self):
        self.assertEqual(
            self._variant_for("HR-2024-07 号文怎么请假", '{"query":"员工请假管理办法"}'), ""
        )
        self.assertEqual(
            self._variant_for("报销超过5000元怎么办", '{"query":"大额报销审批流程"}'), ""
        )
        self.assertEqual(
            self._variant_for("HR-2024-07 号文怎么请假", '{"query":"HR-2024-07 号文规定的请假流程"}'),
            "HR-2024-07 号文规定的请假流程",
        )

    def test_query_variant_rejects_an_over_long_expansion(self):
        self.assertEqual(self._variant_for("年假", '{"query":"' + "年假制度" * 30 + '"}'), "")

    def test_query_variant_failure_or_timeout_means_no_variant(self):
        llm = AsyncMock()
        llm.ainvoke.side_effect = RuntimeError("接口不可用")
        with patch.object(chat_rewrite, "create_rewrite_llm", return_value=llm):
            self.assertEqual(
                asyncio.run(chat_rewrite.build_query_variant("年假几天", self._MODEL)), ""
            )

        async def never_returns(_messages):
            await asyncio.sleep(10)

        llm = SimpleNamespace(ainvoke=never_returns)
        with patch.object(chat_rewrite, "create_rewrite_llm", return_value=llm), patch.object(
            chat_rewrite, "QUERY_VARIANT_TIMEOUT_SECONDS", 0.01
        ):
            self.assertEqual(
                asyncio.run(chat_rewrite.build_query_variant("年假几天", self._MODEL)), ""
            )

    def test_plan_query_variant_is_skipped_when_disabled_or_for_images(self):
        from services.chat.diagnostics import RetrievalDiagnostics

        for enabled, image in ((False, None), (True, "base64-image")):
            diagnostics = RetrievalDiagnostics("问题")
            with patch.object(
                chat_service.application_config, "chat_query_variant_enabled", return_value=enabled
            ), patch.object(
                chat_service, "build_query_variant", new=AsyncMock(return_value="补充")
            ) as builder:
                variant = asyncio.run(
                    chat_service._plan_query_variant("问题", self._MODEL, image, diagnostics)
                )
            builder.assert_not_called()
            self.assertEqual(variant, "")
            self.assertNotIn("query_variant", diagnostics.elapsed_ms)

    def test_plan_query_variant_records_the_variant_when_enabled(self):
        from services.chat.diagnostics import RetrievalDiagnostics

        diagnostics = RetrievalDiagnostics("请假要啥手续")
        with patch.object(
            chat_service.application_config, "chat_query_variant_enabled", return_value=True
        ), patch.object(
            chat_service, "build_query_variant", new=AsyncMock(return_value="员工请假流程")
        ):
            variant = asyncio.run(
                chat_service._plan_query_variant("请假要啥手续", self._MODEL, None, diagnostics)
            )
        self.assertEqual(variant, "员工请假流程")
        self.assertEqual(diagnostics.variant_query, "员工请假流程")
        self.assertIn("query_variant", diagnostics.elapsed_ms)

    @staticmethod
    def _doc(text, source="手册.pdf"):
        return Document(page_content=text, metadata={"source": source})

    def test_variant_candidates_are_interleaved_and_the_primary_query_is_kept(self):
        from services.chat.diagnostics import RetrievalDiagnostics

        primary = [self._doc("主一"), self._doc("主二")]
        variant = [self._doc("补一"), self._doc("补二")]
        retrieve = AsyncMock(side_effect=lambda _kb, query, *_a, **_k: primary if query == "原问题" else variant)
        diagnostics = RetrievalDiagnostics("原问题")

        with patch.object(chat_query, "retrieve_rerank_candidates", retrieve), patch.object(
            chat_query.retrieval_service, "deduplicate_chunk_routes", side_effect=lambda docs: docs
        ):
            documents, used_query = asyncio.run(
                chat_query.retrieve_candidates_with_query_fallback(
                    "技术部", "原问题", "原问题", False,
                    variant_query="书面问题", diagnostics=diagnostics,
                )
            )

        self.assertEqual(used_query, "原问题")
        self.assertEqual([d.page_content for d in documents], ["主一", "补一", "主二", "补二"])
        self.assertEqual(diagnostics.counts["variant_candidates"], 2)
        self.assertEqual(retrieve.await_count, 2)

    def test_variant_retrieval_failure_does_not_affect_the_primary_result(self):
        primary = [self._doc("主一")]

        async def retrieve(_kb, query, *_args, **_kwargs):
            if query == "书面问题":
                raise RuntimeError("补充检索失败")
            return primary

        with patch.object(chat_query, "retrieve_rerank_candidates", side_effect=retrieve):
            documents, used_query = asyncio.run(
                chat_query.retrieve_candidates_with_query_fallback(
                    "技术部", "原问题", "原问题", False, variant_query="书面问题"
                )
            )

        self.assertEqual(documents, primary)
        self.assertEqual(used_query, "原问题")

    def test_variant_equal_to_the_primary_query_does_not_trigger_a_second_retrieval(self):
        retrieve = AsyncMock(return_value=[self._doc("主一")])
        with patch.object(chat_query, "retrieve_rerank_candidates", retrieve):
            asyncio.run(
                chat_query.retrieve_candidates_with_query_fallback(
                    "技术部", "原问题", "原问题", False, variant_query="原问题"
                )
            )
        retrieve.assert_awaited_once()

    def test_variant_candidates_still_fall_back_to_the_original_question_when_empty(self):
        fallback = [self._doc("原问题命中")]

        async def retrieve(_kb, query, *_args, **_kwargs):
            return fallback if query == "原问题" else []

        with patch.object(chat_query, "retrieve_rerank_candidates", side_effect=retrieve):
            documents, used_query = asyncio.run(
                chat_query.retrieve_candidates_with_query_fallback(
                    "技术部", "原问题", "改写后的问题", False, variant_query="书面问题"
                )
            )

        self.assertEqual(documents, fallback)
        self.assertEqual(used_query, "原问题")

    # ------------------------------------------------------------------
    # 工具化检索决策
    # ------------------------------------------------------------------

    @staticmethod
    def _planner_llm(response):
        """构造一个 bind_tools 后返回给定响应的假模型工厂。"""
        bound = AsyncMock()
        bound.ainvoke.return_value = response
        llm = SimpleNamespace(bind_tools=lambda tools: bound)
        return llm, bound

    def test_planner_returns_the_rewritten_query_from_a_tool_call(self):
        llm, bound = self._planner_llm(
            SimpleNamespace(
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {"query": "乙型号的保修期多久"},
                    }
                ],
                additional_kwargs={},
            )
        )
        with patch.object(chat_planner, "create_planner_llm", return_value=llm):
            decision = asyncio.run(
                chat_planner.plan_retrieval(
                    "乙型号呢？",
                    [("user", "甲型号的保修期多久？")],
                    {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
                )
            )

        self.assertTrue(decision.should_search)
        self.assertEqual(decision.query, "乙型号的保修期多久")
        self.assertIn("甲型号的保修期多久", bound.ainvoke.await_args.args[0][1].content)

    def test_planner_reads_tool_calls_from_additional_kwargs(self):
        llm, _ = self._planner_llm(
            SimpleNamespace(
                tool_calls=[],
                additional_kwargs={
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_knowledge_base",
                                "arguments": '{"query": "年假天数规定"}',
                            }
                        }
                    ]
                },
            )
        )
        with patch.object(chat_planner, "create_planner_llm", return_value=llm):
            decision = asyncio.run(
                chat_planner.plan_retrieval(
                    "年假有几天",
                    [],
                    {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
                )
            )

        self.assertTrue(decision.should_search)
        self.assertEqual(decision.query, "年假天数规定")

    def test_planner_skips_retrieval_for_an_explicit_direct_answer_call(self):
        llm, _ = self._planner_llm(
            SimpleNamespace(
                tool_calls=[{"name": "respond_without_search", "args": {}}],
                additional_kwargs={},
            )
        )
        with patch.object(chat_planner, "create_planner_llm", return_value=llm):
            decision = asyncio.run(
                chat_planner.plan_retrieval(
                    "你好呀",
                    [],
                    {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
                )
            )

        self.assertFalse(decision.should_search)

    def test_planner_returns_none_when_tools_are_silently_ignored(self):
        llm, _ = self._planner_llm(
            SimpleNamespace(tool_calls=[], additional_kwargs={}, content="年假有五天。")
        )
        with patch.object(chat_planner, "create_planner_llm", return_value=llm):
            decision = asyncio.run(
                chat_planner.plan_retrieval(
                    "年假有几天",
                    [],
                    {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
                )
            )

        self.assertIsNone(decision)

    def test_planner_falls_back_to_the_original_question_on_unusable_arguments(self):
        llm, _ = self._planner_llm(
            SimpleNamespace(
                tool_calls=[{"name": "search_knowledge_base", "args": {"query": "  "}}],
                additional_kwargs={},
            )
        )
        with patch.object(chat_planner, "create_planner_llm", return_value=llm):
            decision = asyncio.run(
                chat_planner.plan_retrieval(
                    "年假有几天",
                    [],
                    {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
                )
            )

        self.assertTrue(decision.should_search)
        self.assertEqual(decision.query, "年假有几天")

    def test_planner_returns_none_when_the_endpoint_rejects_tools(self):
        bound = AsyncMock()
        bound.ainvoke.side_effect = RuntimeError("tools not supported")
        llm = SimpleNamespace(bind_tools=lambda tools: bound)
        with patch.object(chat_planner, "create_planner_llm", return_value=llm):
            decision = asyncio.run(
                chat_planner.plan_retrieval(
                    "年假有几天",
                    [],
                    {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
                )
            )

        self.assertIsNone(decision)

    def test_plan_query_falls_back_to_rewrite_when_planning_is_disabled(self):
        plan = chat_rewrite.RetrievalQueryPlan(
            query="改写后的问题", used_history=True, inherited_focus="保修期"
        )
        with patch.object(
            chat_service.get_application_container().knowledge_base_settings,
            "query_rewrite_enabled",
            return_value=True,
        ), patch.object(
            chat_service.application_config,
            "chat_retrieval_planning_enabled",
            return_value=False,
        ), patch.object(
            chat_service, "plan_retrieval", new=AsyncMock()
        ) as planner, patch.object(
            chat_service, "build_retrieval_query_plan", new=AsyncMock(return_value=plan)
        ):
            resolved_plan, should_search = asyncio.run(
                chat_service._plan_query(
                    "技术部",
                    "乙型号呢？",
                    [("user", "甲型号的保修期多久？")],
                    "",
                    {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
                    None,
                )
            )

        planner.assert_not_called()
        self.assertTrue(should_search)
        self.assertEqual(resolved_plan, plan)

    def test_plan_query_falls_back_to_rewrite_when_planning_is_unavailable(self):
        plan = chat_rewrite.RetrievalQueryPlan(query="改写后的问题", used_history=True)
        with patch.object(
            chat_service.get_application_container().knowledge_base_settings,
            "query_rewrite_enabled",
            return_value=True,
        ), patch.object(
            chat_service.application_config,
            "chat_retrieval_planning_enabled",
            return_value=True,
        ), patch.object(
            chat_service, "plan_retrieval", new=AsyncMock(return_value=None)
        ), patch.object(
            chat_service, "build_retrieval_query_plan", new=AsyncMock(return_value=plan)
        ):
            resolved_plan, should_search = asyncio.run(
                chat_service._plan_query(
                    "技术部",
                    "乙型号呢？",
                    [],
                    "",
                    {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
                    None,
                )
            )

        self.assertTrue(should_search)
        self.assertEqual(resolved_plan, plan)

    def test_plan_query_does_not_consult_the_planner_for_image_requests(self):
        with patch.object(
            chat_service.get_application_container().knowledge_base_settings,
            "query_rewrite_enabled",
            return_value=True,
        ), patch.object(
            chat_service.application_config,
            "chat_retrieval_planning_enabled",
            return_value=True,
        ), patch.object(
            chat_service, "plan_retrieval", new=AsyncMock()
        ) as planner, patch.object(
            chat_service, "build_retrieval_query_plan", new=AsyncMock()
        ) as rewriter:
            resolved_plan, should_search = asyncio.run(
                chat_service._plan_query(
                    "技术部",
                    "这张图说明什么",
                    [],
                    "",
                    {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
                    "base64-image",
                )
            )

        planner.assert_not_called()
        rewriter.assert_not_called()
        self.assertTrue(should_search)
        self.assertEqual(resolved_plan, chat_rewrite.original_query_plan("这张图说明什么"))

    def test_direct_answer_prompt_omits_the_document_section(self):
        messages, fallback_messages = chat_service._build_llm_messages(
            "谢谢你",
            "谢谢你",
            "",
            None,
            None,
            "",
            [],
            retrieval_used=False,
        )

        self.assertIsNone(fallback_messages)
        self.assertNotIn("[文档内容]", messages[-1].content)
        self.assertEqual(messages[-1].content, "[问题]: 谢谢你")
        self.assertIn("无需查阅知识库", messages[0].content)

    def test_skipped_retrieval_answers_directly_instead_of_reporting_no_evidence(self):
        class DirectStreamingModel:
            def __init__(self):
                self.messages = []

            async def astream(self, messages):
                self.messages = messages
                yield SimpleNamespace(content="不客气。")

        async def collect():
            return [
                event
                async for event in chat_service._stream_answer(
                    "session-direct",
                    {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
                    [],
                    [chat_service.HumanMessage(content="[问题]: 谢谢你")],
                    None,
                    None,
                    allow_answer_without_evidence=True,
                )
            ]

        model = DirectStreamingModel()
        with patch.object(
            chat_service, "create_answer_llm", return_value=model
        ), patch.object(chat_service.chat_sessions, "save_assistant_message") as save:
            events = asyncio.run(collect())

        chunks = [
            json.loads(event.removeprefix("data: ").strip()) for event in events
        ]
        self.assertEqual(chunks[0]["chunk"], "不客气。")
        self.assertEqual(chunks[-1]["final_answer"], "不客气。")
        self.assertNotIn(chat_service.NO_EVIDENCE_ANSWER, str(chunks))
        self.assertTrue(model.messages)
        save.assert_called_once()

    def _run_planned_chat(self, question, tool_calls):
        """跑完整 chat()，只在 LLM 边界打桩，返回 (检索 mock, 回答模型)。"""
        bound = AsyncMock()
        bound.ainvoke.return_value = SimpleNamespace(
            tool_calls=tool_calls, additional_kwargs={}
        )
        planner_llm = SimpleNamespace(bind_tools=lambda tools: bound)

        class PlannedStreamingModel:
            def __init__(self):
                self.messages = []

            async def astream(self, messages):
                self.messages = messages
                yield SimpleNamespace(content="回答内容")

        model = PlannedStreamingModel()
        request = chat_service.ChatRequest(question=question, kb_name="技术部")

        self.last_events = []

        async def drive():
            stream = await chat_service.chat(request, self.owner)
            async for chunk in stream.events:
                self.last_events.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

        with patch.object(
            chat_service,
            "select_model_config",
            return_value={"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_service.get_application_container().knowledge_base_settings,
            "query_rewrite_enabled",
            return_value=True,
        ), patch.object(
            chat_service.application_config,
            "chat_retrieval_planning_enabled",
            return_value=True,
        ), patch.object(
            chat_planner, "create_planner_llm", return_value=planner_llm
        ), patch.object(
            chat_service,
            "retrieve_candidates_with_query_fallback",
            new=AsyncMock(return_value=([], "")),
        ) as retrieve, patch.object(
            chat_service, "create_answer_llm", return_value=model
        ):
            asyncio.run(drive())
        return retrieve, model

    def test_chat_skips_retrieval_entirely_for_small_talk(self):
        retrieve, model = self._run_planned_chat(
            "你好呀", [{"name": "respond_without_search", "args": {}}]
        )

        retrieve.assert_not_called()
        self.assertIn("无需查阅知识库", model.messages[0].content)

    def test_chat_still_retrieves_with_the_query_chosen_by_the_planner(self):
        retrieve, model = self._run_planned_chat(
            "年假有几天",
            [{"name": "search_knowledge_base", "args": {"query": "年假天数规定"}}],
        )

        retrieve.assert_called_once()
        self.assertEqual(retrieve.await_args.args[2], "年假天数规定")
        # 候选为空时走"未找到依据"分支，不会调用回答模型。
        self.assertEqual(model.messages, [])

    def _end_event(self, events):
        return json.loads(next(event[6:] for event in events if '"is_end": true' in event))

    def test_skipped_retrieval_reports_diagnostics_without_search_facts(self):
        self._run_planned_chat("你好呀", [{"name": "respond_without_search", "args": {}}])

        diagnostics = self._end_event(self.last_events)["diagnostics"]
        self.assertTrue(diagnostics["skipped"])
        self.assertEqual(diagnostics["original_question"], "你好呀")
        self.assertEqual(diagnostics["rerank_status"], "not_run")
        self.assertEqual(diagnostics["counts"], {})
        self.assertEqual(diagnostics["elapsed_ms"], {})
        self.assertEqual(diagnostics["context_sources"], [])

    def test_no_evidence_answer_still_carries_diagnostics(self):
        request = chat_service.ChatRequest(question="没有依据的问题", kb_name="技术部")

        async def collect(stream):
            return [c.decode() if isinstance(c, bytes) else c async for c in stream.events]

        with patch.object(
            chat_service, "select_model_config",
            return_value={"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_query, "retrieve_rerank_candidates", new=AsyncMock(return_value=[])
        ), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=[]
        ):
            stream = asyncio.run(chat_service.chat(request, self.owner))
            events = asyncio.run(collect(stream))

        end_event = self._end_event(events)
        self.assertEqual(end_event["final_answer"], chat_service.NO_EVIDENCE_ANSWER)
        diagnostics = end_event["diagnostics"]
        self.assertFalse(diagnostics["skipped"])
        self.assertEqual(diagnostics["actual_query"], "没有依据的问题")
        self.assertEqual(diagnostics["rerank_status"], "not_run")
        self.assertEqual(diagnostics["counts"]["context_spans"], 0)
        self.assertEqual(diagnostics["context_sources"], [])
        self.assertEqual(diagnostics["ranking"], [])

    def test_diagnostics_report_context_routes_separately_from_cited_sources(self):
        class CitingModel:
            async def astream(self, _messages):
                yield SimpleNamespace(content="结论。[来源 2]")

        spans = [
            context_builder.ContextSpan(
                source="手册.pdf", content="第一条上下文", pages=("0",), page_unit="页",
                rank=0, kb_name="技术部", chunk_routes=("fixed",),
            ),
            context_builder.ContextSpan(
                source="手册.pdf", content="第二条上下文，包含结论依据。", pages=("3",),
                page_unit="页", rank=1, kb_name="技术部", chunk_routes=("fixed", "structured"),
            ),
        ]
        request = chat_service.ChatRequest(question="结论是什么", kb_name="技术部")

        async def collect(stream):
            return [c.decode() if isinstance(c, bytes) else c async for c in stream.events]

        with patch.object(
            chat_service, "select_model_config",
            return_value={"api_key": "k", "base_url": "https://example.test/v1", "model": "m"},
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_query, "retrieve_rerank_candidates", new=AsyncMock(return_value=[])
        ), patch.object(chat_service, "rerank", new=AsyncMock(return_value=[])), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=spans
        ), patch.object(chat_service, "create_answer_llm", return_value=CitingModel()):
            stream = asyncio.run(chat_service.chat(request, self.owner))
            events = asyncio.run(collect(stream))

        end_event = self._end_event(events)
        # 送给模型的证据包含两条；回答实际引用的只有第 2 条。
        self.assertEqual([item["id"] for item in end_event["diagnostics"]["context_sources"]], [1, 2])
        self.assertEqual(
            [item["routes"] for item in end_event["diagnostics"]["context_sources"]],
            [["fixed"], ["fixed", "structured"]],
        )
        self.assertEqual([item["id"] for item in end_event["sources"]], [2])
        self.assertEqual(end_event["diagnostics"]["counts"]["context_spans"], 2)
        self.assertNotIn("content", end_event["diagnostics"]["context_sources"][0])

    def test_json_object_parser_handles_nested_objects_and_braces_in_strings(self):
        nested = '前缀 {"query": "a", "meta": {"k": 1}} 后缀'
        self.assertEqual(
            chat_rewrite._json_object_from_response(nested),
            {"query": "a", "meta": {"k": 1}},
        )
        braces = '```json\n{"query": "编号 {A-1} 的规格"}\n```'
        self.assertEqual(
            chat_rewrite._json_object_from_response(braces),
            {"query": "编号 {A-1} 的规格"},
        )
        self.assertIsNone(chat_rewrite._json_object_from_response("没有 JSON"))
        self.assertIsNone(chat_rewrite._json_object_from_response("{坏的"))

    def test_answer_llm_does_not_penalize_repeated_tokens(self):
        # 频率/存在惩罚会压低重复出现的编号、数字和 [来源 N]，且部分兼容接口会拒绝这些参数。
        with patch.object(model_config, "_create_chat_openai") as create:
            model_config.create_answer_llm({"base_url": "https://example.test/v1", "model": "m"})
        kwargs = create.call_args.kwargs
        self.assertNotIn("frequency_penalty", kwargs)
        self.assertNotIn("presence_penalty", kwargs)


if __name__ == "__main__":
    unittest.main()
