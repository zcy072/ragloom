import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

try:
    from langchain_core.documents import Document

    from services.chat import agentic
    from services.chat.diagnostics import RetrievalDiagnostics
except ModuleNotFoundError:
    Document = None
    agentic = None

MODEL = {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"}


def _doc(text, source="手册.pdf"):
    return Document(page_content=text, metadata={"source": source})


def _call(name, **args):
    return SimpleNamespace(tool_calls=[{"name": name, "args": args}], additional_kwargs={})


@unittest.skipIf(agentic is None, "backend dependencies are not installed")
class AgenticRetrievalTests(unittest.TestCase):
    def test_follow_up_query_requires_a_new_search_again_call(self):
        tried = {"甲型号保修期"}
        self.assertEqual(
            agentic._follow_up_query(_call("search_again", query="乙型号保修期"), tried),
            "乙型号保修期",
        )
        self.assertIsNone(
            agentic._follow_up_query(_call("search_again", query="甲型号保修期"), tried)
        )
        self.assertIsNone(agentic._follow_up_query(_call("evidence_sufficient"), tried))
        self.assertIsNone(
            agentic._follow_up_query(SimpleNamespace(tool_calls=[], additional_kwargs={}), tried)
        )

    def test_next_query_is_none_when_the_model_call_fails(self):
        with patch.object(agentic, "create_planner_llm", side_effect=RuntimeError("boom")):
            self.assertIsNone(asyncio.run(agentic.next_query("问题", [_doc("a")], {"问题"}, MODEL)))

    def test_merge_evidence_keeps_the_original_top_ranks_in_place(self):
        current = [_doc(f"o{i}") for i in range(agentic.RERANK_TOP_N)]
        merged = agentic._merge_evidence(current, [_doc("o1"), _doc("n1"), _doc("n2")])
        texts = [doc.page_content for doc in merged]
        self.assertEqual(texts[:6], [f"o{i}" for i in range(6)])
        self.assertEqual(texts[6:8], ["n1", "n2"])
        self.assertEqual(len(texts), agentic.RERANK_TOP_N)

    def test_refine_evidence_stops_when_evidence_is_sufficient(self):
        diagnostics = RetrievalDiagnostics("问题")
        docs = [_doc("a")]
        with patch.object(agentic, "next_query", new=AsyncMock(return_value=None)), patch.object(
            agentic, "retrieve_candidates_with_query_fallback", new=AsyncMock()
        ) as retrieve:
            result = asyncio.run(
                agentic.refine_evidence(
                    "kb", "问题", "问题", docs, MODEL, None, container=None, diagnostics=diagnostics
                )
            )
        retrieve.assert_not_called()
        self.assertEqual(result, docs)
        self.assertEqual(diagnostics.counts["agentic_rounds"], 0)

    def test_refine_evidence_is_bounded_and_merges_new_evidence(self):
        diagnostics = RetrievalDiagnostics("问题")
        queries = iter(["q2", "q3", "q4"])
        with patch.object(
            agentic, "next_query", new=AsyncMock(side_effect=lambda *a: next(queries))
        ), patch.object(
            agentic,
            "retrieve_candidates_with_query_fallback",
            new=AsyncMock(side_effect=lambda kb, q, *a, **k: ([_doc(q)], q)),
        ) as retrieve, patch.object(
            agentic, "rerank", new=AsyncMock(side_effect=lambda q, docs, **k: docs)
        ):
            result = asyncio.run(
                agentic.refine_evidence(
                    "kb", "问题", "问题", [_doc("a")], MODEL, None,
                    container=None, diagnostics=diagnostics,
                )
            )
        self.assertEqual(retrieve.await_count, 2)
        self.assertEqual(diagnostics.counts["agentic_rounds"], 2)
        self.assertEqual({doc.page_content for doc in result}, {"a", "q2", "q3"})

    def test_refine_evidence_keeps_existing_evidence_when_follow_up_fails(self):
        diagnostics = RetrievalDiagnostics("问题")
        docs = [_doc("a")]
        with patch.object(agentic, "next_query", new=AsyncMock(return_value="q2")), patch.object(
            agentic,
            "retrieve_candidates_with_query_fallback",
            new=AsyncMock(side_effect=RuntimeError("down")),
        ):
            result = asyncio.run(
                agentic.refine_evidence(
                    "kb", "问题", "问题", docs, MODEL, None, container=None, diagnostics=diagnostics
                )
            )
        self.assertEqual(result, docs)
        self.assertEqual(diagnostics.counts["agentic_rounds"], 0)


if __name__ == "__main__":
    unittest.main()
