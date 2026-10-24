import unittest

from pydantic import TypeAdapter

from services.chat.contracts import RetrievalDiagnosticPayload
from services.chat.diagnostics import RetrievalDiagnostics


class DiagnosticsTest(unittest.TestCase):
    def test_bounds_and_omits_evidence_body(self):
        diagnostic = RetrievalDiagnostics("question")
        source = {"id": 1, "source": "file.pdf", "page": "1", "content": "private body"}
        result = diagnostic.payload([source] * 40)
        self.assertEqual(len(result["context_sources"]), 32)
        self.assertNotIn("content", result["context_sources"][0])
        TypeAdapter(RetrievalDiagnosticPayload).validate_python(result)

    def test_invalid_or_oversized_payload_is_optional(self):
        diagnostic = RetrievalDiagnostics("question")
        self.assertEqual(diagnostic.payload([{}]), {})
        self.assertEqual(diagnostic.payload([{
            "id": 1, "source": "x" * 40000, "page": "1",
        }]), {})

    def test_payload_reports_the_variant_query_within_the_contract(self):
        diagnostic = RetrievalDiagnostics("请假要啥手续", variant_query="员工请假流程")
        result = diagnostic.payload([])
        self.assertEqual(result["variant_query"], "员工请假流程")
        TypeAdapter(RetrievalDiagnosticPayload).validate_python(result)
