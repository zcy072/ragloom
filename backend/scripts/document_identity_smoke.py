"""用临时 SQLite、原文件目录和唯一 Milvus collection 验证文档身份链路。

运行：python scripts/document_identity_smoke.py --milvus-uri http://milvus:19530
仅使用本地文本解析和固定二维向量，不调用模型服务；退出时清理本次 collection。
"""

import argparse
import dataclasses
import io
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milvus-uri", required=True)
    args = parser.parse_args()
    prefix = f"identity_smoke_{uuid.uuid4().hex}_"
    with tempfile.TemporaryDirectory(prefix="ragloom-identity-") as temporary:
        os.environ.update(
            STORAGE_DIR=temporary,
            DATA_DIR=f"{temporary}/data",
            DB_FILE=f"{temporary}/data/app_data.sqlite3",
            MINERU_RESULTS_DIR=f"{temporary}/mineru_results",
            MILVUS_URI=args.milvus_uri,
            MILVUS_COLLECTION_PREFIX=prefix,
            EMBEDDING_DIMENSION="2",
        )
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import app_storage
        from services.chat.sources import build_source_catalog, select_cited_sources
        from services.core.container import get_application_container
        from services.documents import use_cases
        from services.indexing import pipeline, worker
        from services.retrieval import context, retriever
        from services.retrieval.chunk_index import get_chunk_index

        class Embeddings:
            def embed_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, _query):
                return [1.0, 0.0]

        app_storage.initialize_database()
        kb_id = app_storage.knowledge_bases.create("身份集成验证", 800, 150).kb_id
        container = dataclasses.replace(
            get_application_container(),
            create_embeddings=Embeddings,
            create_query_embeddings=Embeddings,
        )
        index = get_chunk_index()
        try:
            index.ensure_collection()
            results = []
            with patch.object(worker, "enqueue", return_value=False):
                for text in ("审批需要负责人确认。", "审批需要财务负责人复核。"):
                    result = use_cases.upload_document(
                        kb_id,
                        "同名.txt",
                        io.BytesIO(text.encode()),
                        metadata={},
                        parser_mode="native_text",
                    )
                    results.append(result)
                    pipeline.run_document_job(
                        result.job_id, kb_id, result.document_id, container=container
                    )
                first, second = results
                expected = {first.document_id, second.document_id}
                hits = retriever.build_retriever(kb_id, container=container).invoke("审批")
                assert {hit.metadata["document_id"] for hit in hits} == expected
                spans = context.build_context_spans(kb_id, hits)
                citations = select_cited_sources(
                    "两份依据[来源 1][来源 2]", build_source_catalog(spans)
                )
                assert {source["document_id"] for source in citations} == expected
                refreshed = use_cases.reindex_document(kb_id, first.document_id)
                pipeline.run_document_job(
                    refreshed.job_id, kb_id, first.document_id, container=container
                )
                deletion = use_cases.delete_document(kb_id, first.document_id)
                pipeline.run_delete_job(
                    deletion.job_id, kb_id, first.document_id, container=container
                )
                assert index.search_lexical(kb_id, "审批", None, [first.document_id], 10) == []
                assert index.search_dense(kb_id, [1.0, 0.0], None, [first.document_id], 10) == []
                surviving = retriever.build_retriever(kb_id, container=container).invoke("审批")
                assert {hit.metadata["document_id"] for hit in surviving} == {second.document_id}
                assert Path(
                    app_storage.documents.document_path(kb_id, second.document_id)
                ).is_file()
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "collection": index._collection,
                        "checks": [
                            "same_name_upload",
                            "hybrid_search",
                            "parent_restore",
                            "citations",
                            "reindex",
                            "delete_isolation",
                        ],
                    }
                )
            )
        finally:
            # prefix 只在本次运行生成，绝不接收业务 collection 名称。
            assert index._collection.startswith(prefix)
            client = index._get_client()
            if client.has_collection(index._collection):
                client.drop_collection(index._collection)


if __name__ == "__main__":
    main()
