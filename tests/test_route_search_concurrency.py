import threading
import unittest
from unittest.mock import MagicMock

from langchain_core.documents import Document as LCDocument

from services.retrieval import retriever as retrieval_service


def _doc(source: str, index: int) -> LCDocument:
    return LCDocument(page_content=source, metadata={"source": source, "chunk_index": index})


class RunSearchesTests(unittest.TestCase):
    def test_returns_results_in_input_order(self):
        barrier = threading.Barrier(3, timeout=5)

        def make(value):
            def call():
                # 三个调用必须同时在线程里，证明确实并发执行。
                barrier.wait()
                return [value]

            return call

        results = retrieval_service._run_searches([make("a"), make("b"), make("c")])

        self.assertEqual(results, [["a"], ["b"], ["c"]])

    def test_failure_propagates(self):
        def boom():
            raise RuntimeError("milvus down")

        with self.assertRaisesRegex(RuntimeError, "milvus down"):
            retrieval_service._run_searches([lambda: [], boom])

    def test_empty_and_single_call_do_not_use_pool(self):
        self.assertEqual(retrieval_service._run_searches([]), [])
        self.assertEqual(retrieval_service._run_searches([lambda: ["x"]]), [["x"]])


class WeightedRouteRetrieverTests(unittest.TestCase):
    def test_fusion_uses_plan_order_and_skips_empty_allowlists(self):
        index = MagicMock()
        index.search_hybrid.side_effect = lambda kb_id, *_args: {
            "kb1": [_doc("a", 0), _doc("b", 1)],
            "kb2": [_doc("b", 1)],
        }[kb_id]
        embeddings = MagicMock()
        embeddings.embed_query.return_value = [0.1]
        plans = [
            retrieval_service._RoutePlan("one", "kb1", "fixed", 0.5, None),
            retrieval_service._RoutePlan("two", "kb2", "fixed", 0.5, None),
            retrieval_service._RoutePlan("empty", "kb3", "fixed", 0.5, None),
        ]
        retriever = retrieval_service.WeightedRouteRetriever(index, embeddings, plans)

        def allowlist(kb_name, _requested):
            return [] if kb_name == "empty" else [f"{kb_name}.pdf"]

        original = retrieval_service.source_state.current_source_allowlist
        retrieval_service.source_state.current_source_allowlist = allowlist
        try:
            documents = retriever.invoke("q")
        finally:
            retrieval_service.source_state.current_source_allowlist = original

        self.assertEqual(index.search_hybrid.call_count, 2)
        self.assertEqual([d.metadata["source"] for d in documents], ["a", "b", "b"])
        self.assertEqual({d.metadata["kb_name"] for d in documents}, {"one", "two"})


if __name__ == "__main__":
    unittest.main()
