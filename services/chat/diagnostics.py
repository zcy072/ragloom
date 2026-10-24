"""Bounded, optional facts about a single retrieval request."""

import json
from dataclasses import dataclass, field

from services.chat.contracts import RankingDiagnostic, RetrievalDiagnosticPayload


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

    def payload(self, sources: list) -> dict:
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
                "context_sources": [{"id": source["id"], "source": source["source"],
                                     "kb_name": source.get("kb_name", ""), "page": source["page"],
                                     "routes": self.context_routes.get(source["id"], [])}
                                    for source in sources[:32]],
            }
            return result if len(json.dumps(result).encode()) <= 32768 else {}
        except Exception:
            return {}
