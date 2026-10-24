"""Coordination for operations that must update both FTS5 and Chroma."""

import logging
import os

from services.retrieval import chroma as chroma_index
from services.retrieval import fts as fts_index

logger = logging.getLogger(__name__)


def rename_knowledge_base_indexes(old_name: str, new_name: str) -> None:
    """知识库改名后迁移本地索引目录，并同步更新 Chroma 元数据。"""
    old_dir = fts_index.vector_store_dir(old_name)
    new_dir = fts_index.vector_store_dir(new_name)
    fts_moved = False
    if os.path.exists(old_dir):
        if os.path.exists(new_dir):
            raise FileExistsError(f"目标索引目录已存在: {new_name}")
        os.rename(old_dir, new_dir)
        fts_moved = True

    try:
        chroma_index.rename_chroma_knowledge_base(old_name, new_name)
    except Exception as exc:
        # A successful API response must not leave FTS under the new name while Chroma still
        # uses the old metadata. Restore the FTS directory and let the caller roll back the
        # remaining knowledge-base artifacts.
        if fts_moved:
            try:
                os.rename(new_dir, old_dir)
            except OSError as rollback_exc:
                logger.error("FTS rename rollback failed: %s", rollback_exc)
                raise RuntimeError(
                    f"Chroma rename failed ({exc}); FTS rollback also failed ({rollback_exc})"
                ) from exc
        raise
