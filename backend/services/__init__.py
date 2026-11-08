"""Application service modules."""
"""Backend service layer, grouped by business responsibility.

Recommended reading order for the main RAG path:

1. ``documents`` turns uploaded files into normalized documents.
2. ``indexing`` chunks those documents and writes the Milvus chunk index.
3. ``retrieval`` recalls evidence and restores answer-sized context.
4. ``chat`` rewrites follow-ups, reranks evidence, and streams the answer.

``core`` owns application configuration, schema startup and the shared provider-config
boundary, and ``identity`` owns authentication/authorization.
Keeping these dependencies explicit prevents parsing, retrieval, and chat code from
reaching sideways into unrelated implementation details.
"""
