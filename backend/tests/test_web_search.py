import asyncio
import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from langchain_core.documents import Document as LCDocument

from services.chat import service as chat_service
from services.chat import trace as chat_trace
from services.chat import web_search
from services.chat.sources import build_source_catalog, select_cited_sources
from services.retrieval import context as context_builder
from tests import test_chat_trace


def _result(url="https://example.com/a", summary="网页摘要", **extra):
    return {"url": url, "title": "标题", "summary": summary, "site_name": "示例站",
            "date": "2026-10-01T08:00:00+08:00", **extra}


class ParseResultsTest(unittest.TestCase):
    def test_keeps_http_results_with_content_and_drops_the_rest(self):
        results = web_search.parse_results({"results": [
            _result(),
            _result(url="https://example.com/a"),
            _result(url="javascript:alert(1)"),
            _result(url="https://example.com/empty", summary="", snippet=""),
            _result(url="https://example.com/snippet", summary="", snippet="短摘要"),
            "bad",
        ]})
        self.assertEqual([r.url for r in results], ["https://example.com/a", "https://example.com/snippet"])
        self.assertEqual(results[0].date, "2026-10-01")
        self.assertEqual(results[1].content, "短摘要")

    def test_decodes_hex_site_name_and_converts_utc_date_to_local(self):
        with patch.dict(os.environ, {"TZ": "Asia/Shanghai"}):
            time.tzset()
            self.addCleanup(time.tzset)
            results = web_search.parse_results({"results": [
                # 真实接口的返回形式：站点名是 UTF-8 十六进制，日期是 UTC。
                _result(site_name="e5a4aee8a786e7bd91", date="2026-10-05T16:00:00+00:00"),
                _result(url="https://example.com/b", site_name="cafe", date="2026-10-01"),
            ]})
        self.assertEqual((results[0].site_name, results[0].date), ("央视网", "2026-10-06"))
        # 不是合法 UTF-8 的十六进制串保持原样；没有时区的日期不换算。
        self.assertEqual((results[1].site_name, results[1].date), ("cafe", "2026-10-01"))

    def test_rejects_malformed_payload(self):
        with self.assertRaises(web_search.WebSearchError):
            web_search.parse_results({"data": []})


class SearchBudgetTest(unittest.TestCase):
    def setUp(self):
        web_search.budget.reset()
        self.addCleanup(web_search.budget.reset)

    def test_per_user_and_daily_limits(self):
        with patch.dict(os.environ, {"WEB_SEARCH_USER_PER_MINUTE": "2", "WEB_SEARCH_DAILY_LIMIT": "3"}):
            self.assertIsNone(web_search.budget.acquire(1))
            self.assertIsNone(web_search.budget.acquire(1))
            self.assertIn("频繁", web_search.budget.acquire(1))
            self.assertIsNone(web_search.budget.acquire(2))
            self.assertIn("上限", web_search.budget.acquire(3))


class DeepSeekProviderTest(unittest.TestCase):
    def _search(self, handler):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        async def run():
            try:
                return await web_search.DeepSeekSearchProvider("secret").search(
                    "x" * 200, ori_question="原问题", count=6
                )
            finally:
                await client.aclose()

        with patch.object(web_search, "_get_client", return_value=client):
            return asyncio.run(run())

    def test_posts_bounded_query_to_fixed_endpoint(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["Authorization"]
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"results": [_result()]})

        results = self._search(handler)
        self.assertEqual(seen["url"], web_search.DEEPSEEK_SEARCH_URL)
        self.assertEqual(seen["auth"], "Bearer secret")
        self.assertEqual(len(seen["body"]["query"]), web_search.WEB_QUERY_MAX_CHARS)
        self.assertEqual(seen["body"]["ori_question"], "原问题")
        self.assertEqual(len(results), 1)

    def test_maps_http_errors_to_reasons(self):
        with self.assertRaises(web_search.WebSearchError) as raised:
            self._search(lambda _request: httpx.Response(402, json={}))
        self.assertEqual(raised.exception.reason, "账户余额不足")


class SearchWebTest(unittest.TestCase):
    def setUp(self):
        web_search.budget.reset()
        self.addCleanup(web_search.budget.reset)

    def test_missing_key_degrades_without_calling_provider(self):
        with patch.object(web_search, "configured_provider", return_value=None):
            outcome = asyncio.run(web_search.search_web("q", "q", 1))
        self.assertEqual(outcome.documents, [])
        self.assertEqual(outcome.error, "未配置 DeepSeek 密钥")

    def test_provider_failure_degrades(self):
        provider = SimpleNamespace(search=AsyncMock(side_effect=web_search.WebSearchError("搜索超时")))
        outcome = asyncio.run(web_search.search_web("q", "q", 1, provider=provider))
        self.assertEqual((outcome.documents, outcome.error), ([], "搜索超时"))

    def test_failed_search_does_not_consume_quota(self):
        failing = SimpleNamespace(search=AsyncMock(side_effect=RuntimeError("boom")))
        working = SimpleNamespace(search=AsyncMock(return_value=[]))
        with patch.dict(os.environ, {"WEB_SEARCH_USER_PER_MINUTE": "1", "WEB_SEARCH_DAILY_LIMIT": "1"}):
            failed = asyncio.run(web_search.search_web("q", "q", 1, provider=failing))
            succeeded = asyncio.run(web_search.search_web("q", "q", 1, provider=working))
            limited = asyncio.run(web_search.search_web("q", "q", 1, provider=working))
        self.assertEqual(failed.error, "搜索失败：RuntimeError")
        self.assertEqual(succeeded.error, "")
        self.assertIn("上限", limited.error)


class WebSourcesTest(unittest.TestCase):
    def test_web_span_formats_and_catalogues_with_url(self):
        docs = web_search.to_documents(web_search.parse_results({"results": [_result()]}))
        spans = web_search.web_context_spans(docs)
        text = context_builder.format_context_spans(spans)
        self.assertTrue(text.startswith("【来源 1：网页，标题，示例站，2026-10-01，https://example.com/a】"))
        catalog = build_source_catalog(spans, "问题")
        self.assertEqual(catalog[0]["url"], "https://example.com/a")
        self.assertEqual(catalog[0]["evidence"][0]["url"], "https://example.com/a")
        cited = select_cited_sources("答案。[来源 1]", catalog)
        self.assertEqual(cited[0]["url"], "https://example.com/a")


class WebSearchFlowTest(test_chat_trace.ChatFlowTestCase):
    """复用问答追踪的数据库和账号准备，验证联网搜索在完整问答中的行为。"""

    def setUp(self):
        super().setUp()
        web_search.budget.reset()
        self.addCleanup(web_search.budget.reset)

    def test_web_results_are_reranked_cited_and_traced(self):
        request = chat_service.ChatRequest(question="今年的政策？", kb_name="技术部", web_search=True)
        owner = chat_service.SessionOwner(self.plain_id, "user")
        web_doc = LCDocument(page_content="网页正文", metadata={
            "source": "政策解读", "url": "https://example.com/p", "site_name": "政府网",
            "date": "2026-10-01", "kb_name": "", "chunk_route": "web", "relevance_score": 0.9,
        })
        outcome = web_search.WebSearchOutcome([web_doc], 12.0)
        seen_messages = []

        class Model:
            async def astream(self, messages):
                seen_messages.extend(messages)
                yield SimpleNamespace(content="结论。[来源 1]")

        async def rerank(_query, docs, **_kwargs):
            return [doc for doc in docs if doc.metadata.get("url")]

        async def consume():
            stream = await chat_service.chat(request, owner)
            return [event async for event in stream.events]

        with patch.object(chat_service, "select_model_config", return_value={
            "api_key": "k", "base_url": "https://example.test/v1", "model": "m",
        }), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_service, "retrieve_candidates_with_query_fallback",
            new=AsyncMock(return_value=([], "今年的政策？")),
        ), patch.object(chat_service, "rerank", new=rerank), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=[]
        ), patch.object(
            chat_service.web_search, "search_web", new=AsyncMock(return_value=outcome)
        ), patch.object(chat_service, "create_answer_llm", return_value=Model()):
            events = [json.loads(e[6:]) for e in asyncio.run(consume()) if e.startswith("data: ")]

        self.assertIn("https://example.com/p", seen_messages[-1].content)
        final = events[-1]
        self.assertEqual(final["sources"][0]["url"], "https://example.com/p")
        payload = chat_trace.load_trace(final["trace_id"]).payload
        web_stage = next(s for s in payload["stages"] if s["stage"] == "web_search")
        self.assertEqual(web_stage["status"], "ok")
        self.assertEqual(payload["web_results"][0]["url"], "https://example.com/p")
        self.assertTrue(payload["web_results"][0]["kept"])
        self.assertEqual(payload["context"][0]["url"], "https://example.com/p")

    def test_web_search_requires_permission(self):
        (response,) = self._request("plain", "user-password", [
            ("POST", "/api/chat", {"question": "q", "kb_name": "技术部", "web_search": True}),
        ])
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
