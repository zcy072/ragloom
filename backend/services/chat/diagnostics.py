"""Bounded, optional facts about a single retrieval request."""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from services.chat.contracts import ContextDiagnostic, RankingDiagnostic, RetrievalDiagnosticPayload

if TYPE_CHECKING:
    from services.chat.trace import TraceDocument, TraceQuery


@dataclass
class RetrievalDiagnostics:
    original_question: str
    actual_query: str = ""
    variant_query: str = ""
    rewritten: bool = False
    fallback_attempted: bool = False
    skipped: bool = False
    rerank_status: str = "not_run"
    counts: dict[str, int] = field(default_factory=dict)
    elapsed_ms: dict[str, float] = field(default_factory=dict)
    context_routes: dict[int, list[str]] = field(default_factory=dict)
    ranking: list[RankingDiagnostic] = field(default_factory=list)
    # 以下两项只进入持久化的追踪记录，不放进 SSE 诊断事件。
    rerank_results: list["TraceDocument"] = field(default_factory=list)
    agentic_queries: list["TraceQuery"] = field(default_factory=list)

    def payload(self, sources: list) -> RetrievalDiagnosticPayload | dict:
        try:
            result: RetrievalDiagnosticPayload = {
                "original_question": self.original_question[:4000],
                "actual_query": self.actual_query[:4000],
                "variant_query": self.variant_query[:4000],
                "rewritten": self.rewritten,
                "fallback_attempted": self.fallback_attempted,
                "skipped": self.skipped,
                "rerank_status": self.rerank_status,
                "counts": self.counts,
                "elapsed_ms": self.elapsed_ms,
                "ranking": self.ranking[:10],
                "context_sources": self._context_sources(sources),
            }
            return result if len(json.dumps(result).encode()) <= 32768 else {}
        except Exception:
            return {}

    def _context_sources(self, sources: list) -> list[ContextDiagnostic]:
        items: list[ContextDiagnostic] = []
        for source in sources[:32]:
            item: ContextDiagnostic = {
                "id": source["id"],
                "source": source["source"],
                "kb_id": source.get("kb_id", ""),
                "kb_name_snapshot": source.get("kb_name_snapshot", ""),
                "page": source["page"],
                "routes": self.context_routes.get(source["id"], []),
            }
            if source.get("document_id"):
                item["document_id"] = source["document_id"]
            items.append(item)
        return items
