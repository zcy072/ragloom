import os
import sqlite3
import unittest
from unittest.mock import patch

try:
    import app_storage
    from routers import models, schemas
except ModuleNotFoundError:
    app_storage = None

from tests.support import IsolatedStorageTestCase


@unittest.skipUnless(app_storage, "requires the backend Python dependencies")
class ModelConfigTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        self.encryption_key = patch.dict(
            os.environ,
            {"MODEL_API_KEY_ENCRYPTION_KEY": "test-model-key-" + "a" * 32},
            clear=False,
        )
        self.encryption_key.start()
        self.addCleanup(self.encryption_key.stop)
        app_storage.initialize_database()

    def _save(self, provider: str, api_key: str):
        return models.save_model(
            schemas.ModelConfigRequest(
                provider=provider,
                api_key=api_key,
                base_url="https://example.test/v1",
                default_model=f"{provider}-model",
            )
        )

    def test_bootstrap_uses_current_openai_and_deepseek_defaults(self):
        app_storage.bootstrap_defaults()

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            rows = dict(
                conn.execute(
                    "SELECT provider_name, default_model FROM models_config "
                    "WHERE provider_name IN ('OpenAI', 'DeepSeek')"
                ).fetchall()
            )

        self.assertEqual(rows["OpenAI"], "gpt-6-sol")
        self.assertEqual(rows["DeepSeek"], "deepseek-flash")

    def test_first_configured_model_becomes_active_and_can_be_switched(self):
        first = self._save("模型 A", "key-a")
        self.assertTrue(first["is_active"])
        self._save("模型 B", "key-b")

        models.activate_model("模型 B")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            rows = conn.execute(
                "SELECT provider_name, is_active FROM models_config ORDER BY provider_name"
            ).fetchall()
        self.assertEqual(rows, [("模型 A", 0), ("模型 B", 1)])

    def test_saved_key_is_encrypted_at_rest_and_decrypted_for_model_clients(self):
        from services.core import llm_config, model_api_keys

        self._save("模型 A", "sk-sensitive-value")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            stored = conn.execute(
                "SELECT api_key FROM models_config WHERE provider_name = ?",
                ("模型 A",),
            ).fetchone()[0]

        self.assertTrue(model_api_keys.is_encrypted(stored))
        self.assertNotIn("sk-sensitive-value", stored)
        self.assertEqual(
            llm_config.get_model_config("模型 A", app_storage.model_providers)["api_key"],
            "sk-sensitive-value",
        )

    def test_model_clients_preserve_secrets_and_completion_limits(self):
        from pydantic import SecretStr

        from services.core import config, llm_config

        model_config = {
            "api_key": "sk-test-value",
            "base_url": "https://example.test/v1",
            "model": "test-model",
        }
        clients = (
            (llm_config.create_answer_llm(model_config), 4000),
            (llm_config.create_rewrite_llm(model_config), 256),
            (
                llm_config.create_planner_llm(model_config),
                config.CHAT_RETRIEVAL_PLANNING_MAX_TOKENS,
            ),
            (
                llm_config.create_summary_llm(model_config),
                config.CHAT_MEMORY_SUMMARY_MAX_TOKENS,
            ),
        )

        for client, expected_limit in clients:
            self.assertIsInstance(client.openai_api_key, SecretStr)
            self.assertEqual(
                client.openai_api_key.get_secret_value(),
                "sk-test-value",
            )
            self.assertEqual(client.max_tokens, expected_limit)

        with patch.object(
            llm_config,
            "get_siliconflow_api_key",
            return_value="sk-embedding-value",
        ):
            embeddings = llm_config.get_embeddings(app_storage.model_providers)
        self.assertIsInstance(embeddings.openai_api_key, SecretStr)
        self.assertEqual(
            embeddings.openai_api_key.get_secret_value(),
            "sk-embedding-value",
        )

    def test_embeddings_send_raw_text_not_openai_token_ids(self):
        from types import SimpleNamespace

        from services.core import llm_config

        sent = []

        class RecordingClient:
            def create(self, **kwargs):
                sent.append(kwargs["input"])
                count = len(kwargs["input"])
                return SimpleNamespace(
                    model_dump=lambda: {"data": [{"embedding": [0.0, 1.0]} for _ in range(count)]}
                )

        with patch.object(llm_config, "get_siliconflow_api_key", return_value="sk-test"):
            embeddings = llm_config.get_embeddings(app_storage.model_providers)
        embeddings.client = RecordingClient()
        embeddings.embed_query("国家卫健委开展启明行动")
        embeddings.embed_documents(["第一段文本", "第二段文本"])

        # 默认的 tiktoken 预处理会把文本换成 OpenAI 词表的整数 ID，SiliconFlow 无法理解。
        self.assertEqual(sent[0], ["国家卫健委开展启明行动"])
        self.assertEqual(sent[1], ["第一段文本", "第二段文本"])

    def test_embeddings_client_retries_rate_limits_with_bounded_timeout(self):
        from services.core import llm_config

        with patch.object(llm_config, "get_siliconflow_api_key", return_value="sk-test"):
            embeddings = llm_config.get_embeddings(app_storage.model_providers)

        # 并行索引会放大偶发 429，重试次数不能退回库默认的 2 次；超时也不能是默认的 600 秒。
        self.assertEqual(embeddings.max_retries, 5)
        self.assertEqual(embeddings.request_timeout, 120)

    def test_query_embeddings_fail_fast_but_use_the_same_model_as_indexing(self):
        from services.core import llm_config

        with patch.object(llm_config, "get_siliconflow_api_key", return_value="sk-test"):
            indexing = llm_config.get_embeddings(app_storage.model_providers)
            query = llm_config.get_query_embeddings(app_storage.model_providers)

        # 向量必须来自同一模型和同一发送方式，否则查询向量与库内向量不可比。
        self.assertEqual(query.model, indexing.model)
        self.assertEqual(query.openai_api_base, indexing.openai_api_base)
        self.assertFalse(query.check_embedding_ctx_length)
        # 用户在线等待回答：失败要在秒级暴露，而不是像后台索引那样退避重试几分钟。
        self.assertEqual(query.request_timeout, 15)
        self.assertEqual(query.max_retries, 1)

    def test_clearing_active_key_promotes_another_configured_model(self):
        self._save("模型 A", "key-a")
        self._save("模型 B", "key-b")
        models.activate_model("模型 B")
        models.save_model(
            schemas.ModelConfigRequest(
                provider="模型 B",
                api_key="",
                clear_api_key=True,
                base_url="https://example.test/v1",
                default_model="模型 B-model",
            )
        )

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            active = conn.execute(
                "SELECT provider_name FROM models_config WHERE is_active = 1"
            ).fetchone()[0]
        self.assertEqual(active, "模型 A")

    def test_provider_rows_are_mapped_to_public_and_private_contracts(self):
        self._save("模型 A", "key-a")

        state = app_storage.model_providers.check("模型 A")
        public = next(
            item
            for item in app_storage.model_providers.list_public()
            if item["provider"] == "模型 A"
        )
        stored = app_storage.model_providers.get_config("模型 A")

        self.assertEqual(
            state,
            {
                "configured": True,
                "provider": "模型 A",
                "model": "模型 A-model",
                "is_active": True,
            },
        )
        self.assertEqual(public["base_url"], "https://example.test/v1")
        self.assertEqual(public["default_model"], "模型 A-model")
        self.assertTrue(public["configured"])
        self.assertTrue(public["is_active"])
        self.assertIsNotNone(stored)
        self.assertNotEqual(stored.encrypted_api_key, "key-a")
        self.assertEqual(stored.base_url, "https://example.test/v1")
        self.assertEqual(stored.model, "模型 A-model")

if __name__ == "__main__":
    unittest.main()
