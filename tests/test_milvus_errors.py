"""Milvus 不可用时，读路径应抛出明确的 503 类错误；不需要真实 Milvus。"""

import unittest

try:
    from pymilvus.exceptions import MilvusException

    from services.core.errors import ServiceUnavailableError
    from services.retrieval.milvus_index import MilvusChunkIndex
except ModuleNotFoundError:
    MilvusChunkIndex = None


class _DownClient:
    """每个调用都像 Milvus 不可达一样失败。"""

    def has_collection(self, *_args, **_kwargs):
        raise MilvusException(code=2, message="Fail connecting to server")

    def __getattr__(self, _name):
        def fail(*_args, **_kwargs):
            raise MilvusException(code=2, message="Fail connecting to server")

        return fail


@unittest.skipUnless(MilvusChunkIndex, "requires backend dependencies")
class MilvusUnavailableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = MilvusChunkIndex("http://milvus.invalid:19530", "", "test_", 8)
        self.index._client = _DownClient()

    def test_read_paths_raise_service_unavailable(self):
        vector = [0.0] * 8
        calls = {
            "dense": lambda: self.index.search_dense("kb", vector, None, ["a.pdf"], 3),
            "lexical": lambda: self.index.search_lexical("kb", "登革热", None, ["a.pdf"], 3),
            "hybrid": lambda: self.index.search_hybrid(
                "kb", "登革热", vector, None, ["a.pdf"], 3
            ),
            "routes": lambda: self.index.available_routes("kb", ["a.pdf"]),
            "neighbors": lambda: self.index.neighbors("kb", "a.pdf", "fixed", 0, 2),
        }
        for name, call in calls.items():
            with self.subTest(name):
                with self.assertRaises(ServiceUnavailableError) as raised:
                    call()
                self.assertEqual(raised.exception.code, "retrieval_index_unavailable")


if __name__ == "__main__":
    unittest.main()
