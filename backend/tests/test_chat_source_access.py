import unittest

from services.chat import source_access

A, B, C = "kb-a", "kb-b", "kb-c"


def _readable(allowed: set[str], active: set[str] | None = None):
    active = active if active is not None else {A, B, C}
    return source_access.kb_reader(lambda kb: kb in allowed, lambda kb: kb in active)


def _source(kb_id, **extra):
    item = {"id": 1, "source": "x.pdf", "content": f"{kb_id} 原文", **extra}
    if kb_id:
        item["kb_id"] = kb_id
        item["evidence"] = [{"id": 1, "kb_id": kb_id, "content": f"{kb_id} 证据"}]
    return item


class FilterMessageSourcesTest(unittest.TestCase):
    def test_revoked_source_kb_is_removed(self):
        sources = [_source(A), _source(B)]
        result = source_access.filter_message_sources(sources, _readable({A}))
        self.assertEqual([item["kb_id"] for item in result], [A])

    def test_readable_sources_and_web_sources_are_kept(self):
        web = {"id": 3, "source": "页面", "url": "https://example.test", "evidence": [
            {"id": 3, "url": "https://example.test"}]}
        result = source_access.filter_message_sources(
            [_source(A), _source(C), web], _readable({A, C})
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result[2], web)

    def test_deleted_or_deleting_source_kb_is_removed_even_with_permission(self):
        result = source_access.filter_message_sources(
            [_source(A), _source(B)], _readable({A, B}, active={A})
        )
        self.assertEqual([item["kb_id"] for item in result], [A])

    def test_nested_evidence_is_filtered(self):
        merged = _source(A)
        merged["evidence"].append({"id": 2, "kb_id": B, "content": "B 证据"})
        result = source_access.filter_message_sources([merged], _readable({A}))
        self.assertEqual([item["kb_id"] for item in result[0]["evidence"]], [A])

    def test_record_without_kb_id_or_url_is_dropped(self):
        result = source_access.filter_message_sources(
            [{"id": 1, "source": "孤立记录"}], _readable({A})
        )
        self.assertEqual(result, [])


class FilterTracePayloadTest(unittest.TestCase):
    def test_all_source_lists_are_filtered(self):
        def doc(kb_id):
            return {"kb_id": kb_id, "source": "x.pdf", "snippet": f"{kb_id} 片段"}

        web = {"kb_id": "", "source": "页面", "url": "https://example.test", "snippet": "网页"}
        payload = {
            "id": "t", "question": "问题", "answer_note": "保留",
            "candidates": [doc(A), doc(B)],
            "reranked": [doc(B), doc(A)],
            "web_results": [web],
            "context": [doc(A), doc(B)],
        }
        result = source_access.filter_trace_payload(payload, _readable({A}))
        for key in ("candidates", "reranked", "context"):
            self.assertEqual([item["kb_id"] for item in result[key]], [A], key)
        self.assertEqual(result["web_results"], [web])
        self.assertEqual(result["question"], "问题")
        self.assertEqual(payload["candidates"][1]["kb_id"], B)

    def test_permission_is_checked_once_per_kb(self):
        calls = []
        readable = source_access.kb_reader(
            lambda kb: calls.append(kb) or True, lambda kb: True
        )
        source_access.filter_trace_payload(
            {"candidates": [{"kb_id": A}, {"kb_id": A}], "context": [{"kb_id": A}]},
            readable,
        )
        self.assertEqual(calls, [A])


if __name__ == "__main__":
    unittest.main()
