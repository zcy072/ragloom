import asyncio
import os
import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from cryptography.fernet import Fernet
from pydantic import ValidationError

import app_storage
from routers import models, schemas
from services.chat import ranking as chat_ranking
from services.chat import web_search
from services.core import llm_config, model_api_keys
from services.core.container import get_application_container
from services.core.errors import (
    InvalidRequestError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from tests.support import IsolatedStorageAsyncTestCase, IsolatedStorageTestCase, api_client

SILICONFLOW = llm_config.SILICONFLOW_PROVIDER_NAME
CORRUPT_CIPHERTEXT = "enc:v1:corrupted-token"


class ModelConfigTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()

    def _save(
        self,
        provider: str,
        api_key: str = "",
        *,
        base_url: str = "https://example.test/v1",
        clear_api_key: bool = False,
    ):
        return models.save_model(
            schemas.ModelConfigRequest(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                default_model=f"{provider}-model",
                clear_api_key=clear_api_key,
            )
        )

    @staticmethod
    def _stored(provider: str):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            row = conn.execute(
                "SELECT api_key, base_url, is_active FROM models_config WHERE provider_name = ?",
                (provider,),
            ).fetchone()
        return row

    @staticmethod
    def _set_raw_key(provider: str, value: str) -> None:
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE models_config SET api_key = ? WHERE provider_name = ?", (value, provider)
            )

    @staticmethod
    def _active() -> list[str]:
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            return [
                row[0]
                for row in conn.execute(
                    "SELECT provider_name FROM models_config WHERE is_active = 1"
                )
            ]

    def test_saving_a_model_drops_retrievers_holding_the_old_embedding_key(self):
        with patch.object(models.retriever, "clear_retriever_cache") as clear:
            self._save("模型 A", "key-a")

        clear.assert_called_once_with()

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
        self._save("模型 A", "  sk-sensitive-value  ")
        stored = self._stored("模型 A")[0]

        self.assertTrue(stored.startswith(model_api_keys.ENCRYPTED_PREFIX))
        self.assertNotIn("sk-sensitive-value", stored)
        self.assertEqual(
            llm_config.get_model_config("模型 A", app_storage.model_providers)["api_key"],
            "sk-sensitive-value",
        )

    def test_model_clients_preserve_secrets_and_completion_limits(self):
        from pydantic import SecretStr

        from services.core import config

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
        answer_llm = clients[0][0]
        self.assertEqual(answer_llm.max_retries, 1)
        self.assertEqual(answer_llm.request_timeout.read, 90)
        for client, _ in clients:
            self.assertEqual(client.request_timeout.connect, 5)
        for client, _ in clients[1:]:
            self.assertEqual(client.max_retries, 0)

        self._save(SILICONFLOW, "sk-embedding-value")
        embeddings = llm_config.get_embeddings(app_storage.model_providers)
        self.assertIsInstance(embeddings.openai_api_key, SecretStr)
        self.assertEqual(
            embeddings.openai_api_key.get_secret_value(),
            "sk-embedding-value",
        )

    def test_embeddings_send_raw_text_not_openai_token_ids(self):
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
        with patch.object(llm_config, "get_siliconflow_api_key", return_value="sk-test"):
            embeddings = llm_config.get_embeddings(app_storage.model_providers)

        # 并行索引会放大偶发 429，重试次数不能退回库默认的 2 次；超时也不能是默认的 600 秒。
        self.assertEqual(embeddings.max_retries, 5)
        self.assertEqual(embeddings.request_timeout, 120)

    def test_query_embeddings_fail_fast_but_use_the_same_model_as_indexing(self):
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

    def test_embedding_sdk_errors_become_safe_service_errors(self):
        canary = "canary-embedding-7f3e"

        class FailingClient:
            def create(self, **_kwargs):
                raise RuntimeError(f"upstream echoed {canary}")

        self._save(SILICONFLOW, "sk-sf")
        embeddings = llm_config.get_query_embeddings(app_storage.model_providers)
        embeddings.client = FailingClient()
        with self.assertLogs(llm_config.logger, level="WARNING") as logs:
            with self.assertRaises(ServiceUnavailableError) as raised:
                embeddings.embed_query("问题")

        self.assertEqual(raised.exception.code, "embedding_unavailable")
        self.assertIsNone(raised.exception.__cause__)
        self.assertTrue(raised.exception.__suppress_context__)
        self.assertNotIn(canary, str(raised.exception))
        self.assertNotIn(canary, "\n".join(logs.output))

    def test_clearing_active_key_promotes_another_configured_model(self):
        self._save("模型 A", "key-a")
        self._save("模型 B", "key-b")
        models.activate_model("模型 B")
        self._save("模型 B", clear_api_key=True)

        self.assertEqual(self._active(), ["模型 A"])

    def test_save_semantics_keep_replace_clear_and_conflict(self):
        # 新记录不带 Key：保存为未配置。
        self.assertFalse(self._save("模型 A")["is_active"])
        self.assertEqual(self._stored("模型 A")[0], "")

        self._save("模型 A", "key-1")
        first = self._stored("模型 A")[0]
        # 留空保留原密文。
        self._save("模型 A")
        self.assertEqual(self._stored("模型 A")[0], first)
        # 新 Key 替换。
        self._save("模型 A", "key-2")
        second = self._stored("模型 A")[0]
        self.assertNotEqual(second, first)
        self.assertEqual(model_api_keys.decrypt_api_key(second), "key-2")

        # 非空 Key 与清除冲突：拒绝且记录不变。
        with self.assertRaises(InvalidRequestError) as raised:
            self._save("模型 A", "key-3", clear_api_key=True)
        self.assertEqual(raised.exception.code, "invalid_model_configuration")
        self.assertEqual(self._stored("模型 A")[0], second)

        # 显式清除。
        self._save("模型 A", clear_api_key=True)
        self.assertEqual(self._stored("模型 A")[0], "")

    def test_changing_base_url_requires_reentering_or_clearing_the_saved_key(self):
        self._save("模型 A", "key-a")

        with self.assertRaises(InvalidRequestError) as error:
            self._save("模型 A", base_url="https://attacker.test/v1")
        self.assertEqual(error.exception.code, "model_api_key_required")
        self.assertEqual(self._stored("模型 A")[1], "https://example.test/v1")
        # 地址不变时留空仍保留原 key；换地址并填写新 key 或清空都可以保存。
        self._save("模型 A")
        self._save("模型 A", "key-b", base_url="https://other.test/v1")
        self._save("模型 A", base_url="https://third.test/v1", clear_api_key=True)
        self.assertEqual(self._stored("模型 A")[:2], ("", "https://third.test/v1"))

    def test_base_url_must_be_plain_http_or_https(self):
        for base_url in (
            "https://user:pass@example.test/v1",
            "https://example.test/v1?api_key=secret",
            "https://example.test/v1#secret",
            "ftp://example.test/v1",
            "example.test/v1",
            "https:///v1",
            "https://example.test:notaport/v1",
        ):
            with self.subTest(base_url=base_url):
                with self.assertRaises(InvalidRequestError) as raised:
                    self._save("模型 A", "key-a", base_url=base_url)
                self.assertEqual(raised.exception.code, "invalid_model_configuration")
        self.assertIsNone(self._stored("模型 A"))
        # 本地模型可以使用 HTTP。
        self._save("本地", "key", base_url="http://host.docker.internal:11434/v1")

    def test_api_key_length_is_bounded_by_the_request_schema(self):
        with self.assertRaises(ValidationError):
            schemas.ModelConfigRequest(
                provider="模型 A",
                api_key="k" * 4097,
                base_url="https://example.test/v1",
                default_model="m",
            )

    def test_provider_rows_are_mapped_to_public_and_private_contracts(self):
        self._save("模型 A", "key-a")

        state = models.check_model_config("模型 A")
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

    def test_check_contract_is_local_and_never_falls_back(self):
        # 完全没有激活记录。
        self.assertEqual(
            models.check_model_config(None),
            {"configured": False, "provider": None, "model": None, "is_active": False},
        )
        with self.assertRaises(ResourceNotFoundError) as missing:
            models.check_model_config("不存在")
        self.assertEqual(missing.exception.code, "model_provider_not_found")

        self._save("未配置")
        self.assertEqual(
            models.check_model_config("未配置"),
            {"configured": False, "provider": "未配置", "model": "未配置-model", "is_active": False},
        )

        self._save("模型 A", "key-a")
        # 检查只做本地解密，不发任何网络请求。
        with patch("httpx.Client.send") as sync_send, patch(
            "httpx.AsyncClient.send"
        ) as async_send:
            self.assertTrue(models.check_model_config(None)["configured"])
            self.assertTrue(models.check_model_config("模型 A")["is_active"])
        sync_send.assert_not_called()
        async_send.assert_not_called()

    def test_corrupt_ciphertext_is_reported_but_stays_repairable(self):
        self._save("模型 A", "key-a")
        self._save("模型 B", "key-b")
        self._set_raw_key("模型 A", CORRUPT_CIPHERTEXT)

        # 列表不解密，仍显示已保存。
        listed = {item["provider"]: item for item in models.list_models()}
        self.assertTrue(listed["模型 A"]["configured"])
        # 检查与使用都安全报错，不包含密文。
        with self.assertRaises(ServiceUnavailableError) as checked:
            models.check_model_config("模型 A")
        self.assertEqual(checked.exception.code, "model_key_unavailable")
        self.assertNotIn(CORRUPT_CIPHERTEXT, checked.exception.public_message)
        with self.assertRaises(ServiceUnavailableError) as used:
            llm_config.select_model_config("模型 A", app_storage.model_providers)
        self.assertEqual(used.exception.code, "model_key_unavailable")
        # 其他提供商不受影响。
        self.assertEqual(
            llm_config.select_model_config("模型 B", app_storage.model_providers)["api_key"],
            "key-b",
        )

        # 留空保存不解密、不删除坏记录。
        with patch.object(model_api_keys, "decrypt_api_key", side_effect=AssertionError):
            self._save("模型 A")
            self.assertEqual(self._stored("模型 A")[0], CORRUPT_CIPHERTEXT)
            # 直接替换不需要解密旧值。
            self._save("模型 A", "key-new")
        self.assertEqual(
            llm_config.get_model_config("模型 A", app_storage.model_providers)["api_key"],
            "key-new",
        )
        self._set_raw_key("模型 A", CORRUPT_CIPHERTEXT)
        with patch.object(model_api_keys, "decrypt_api_key", side_effect=AssertionError):
            self._save("模型 A", clear_api_key=True)
        self.assertEqual(self._stored("模型 A")[0], "")

    def test_wrong_master_key_blocks_use_not_listing(self):
        self._save("模型 A", "key-a")
        with patch.dict(os.environ, {"MODEL_API_KEY_ENCRYPTION_KEY": Fernet.generate_key().decode()}):
            self.assertTrue(models.list_models()[0]["configured"])
            with self.assertRaises(ServiceUnavailableError) as raised:
                models.check_model_config(None)
            self.assertEqual(raised.exception.code, "model_key_unavailable")
            # 用新主密钥直接替换仍然可行。
            self._save("模型 A", "key-replaced")
            self.assertTrue(models.check_model_config(None)["configured"])

    def test_activation_runs_the_local_check_first(self):
        self._save("模型 A", "key-a")
        self._save("模型 B", "key-b")
        self._set_raw_key("模型 B", CORRUPT_CIPHERTEXT)

        with self.assertRaises(ServiceUnavailableError) as raised:
            models.activate_model("模型 B")
        self.assertEqual(raised.exception.code, "model_key_unavailable")
        self.assertEqual(self._active(), ["模型 A"])

        with self.assertRaises(ResourceNotFoundError):
            models.activate_model("不存在")
        self._save("未配置")
        with self.assertRaises(InvalidRequestError):
            models.activate_model("未配置")
        self.assertEqual(self._active(), ["模型 A"])

    def test_local_placeholders_are_encrypted_and_usable(self):
        app_storage.bootstrap_defaults()

        for provider, placeholder in (("本地 Ollama", "ollama"), ("本地 vLLM", "vllm")):
            with self.subTest(provider=provider):
                stored = self._stored(provider)[0]
                self.assertTrue(stored.startswith(model_api_keys.ENCRYPTED_PREFIX))
                config = llm_config.get_model_config(provider, app_storage.model_providers)
                self.assertEqual(config["api_key"], placeholder)
                client = llm_config.create_answer_llm(config)
                self.assertEqual(client.openai_api_key.get_secret_value(), placeholder)

    def test_environment_key_is_imported_once_and_never_used_as_fallback(self):
        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "sk-from-env"}):
            app_storage.bootstrap_defaults()
            self.assertEqual(self._active(), [SILICONFLOW])
            self.assertEqual(
                llm_config.get_siliconflow_api_key(app_storage.model_providers), "sk-from-env"
            )

            self._save(
                SILICONFLOW,
                clear_api_key=True,
                base_url=self._stored(SILICONFLOW)[1],
            )
            # 模拟重启：默认数据已存在，不会再次导入环境变量。
            app_storage.bootstrap_defaults()
            self.assertEqual(self._stored(SILICONFLOW)[0], "")

            with self.assertRaises(ServiceUnavailableError) as embedding_key:
                llm_config.get_siliconflow_api_key(app_storage.model_providers)
            self.assertEqual(embedding_key.exception.code, "model_not_configured")
            # 本地占位模型已加密保存，清除 SiliconFlow 后被选为新的默认项，而不是回退环境变量。
            active = llm_config.select_model_config(None, app_storage.model_providers)
            self.assertNotEqual(active["api_key"], "sk-from-env")

    def test_model_selection_reports_clear_errors_without_switching(self):
        with self.assertRaises(ServiceUnavailableError) as no_active:
            llm_config.select_model_config(None, app_storage.model_providers)
        self.assertEqual(no_active.exception.code, "model_not_configured")

        self._save("模型 A", "key-a")
        self._save("未配置")
        with self.assertRaises(ResourceNotFoundError) as missing:
            llm_config.select_model_config("不存在", app_storage.model_providers)
        self.assertEqual(missing.exception.code, "model_provider_not_found")
        with self.assertRaises(ServiceUnavailableError) as unconfigured:
            llm_config.select_model_config("未配置", app_storage.model_providers)
        self.assertEqual(unconfigured.exception.code, "model_not_configured")
        self.assertEqual(
            llm_config.select_model_config(None, app_storage.model_providers)["api_key"], "key-a"
        )

    def test_consumers_use_the_decrypted_database_keys(self):
        self._save("模型 A", "sk-chat-synthetic")
        self._save(SILICONFLOW, "sk-sf-synthetic", base_url="https://api.siliconflow.cn/v1")
        self._save(web_search.DEEPSEEK_PROVIDER_NAME, "sk-ds-synthetic",
                   base_url="https://api.deepseek.com/v1")
        models.activate_model("模型 A")
        container = get_application_container()

        chat_config = container.select_model(None)
        for factory in (
            llm_config.create_answer_llm,
            llm_config.create_rewrite_llm,
            llm_config.create_planner_llm,
            llm_config.create_summary_llm,
        ):
            with self.subTest(factory=factory.__name__):
                self.assertEqual(
                    factory(chat_config).openai_api_key.get_secret_value(), "sk-chat-synthetic"
                )
        for embeddings in (container.create_embeddings(), container.create_query_embeddings()):
            self.assertEqual(embeddings.openai_api_key.get_secret_value(), "sk-sf-synthetic")

        _, _, headers = chat_ranking._rerank_endpoint(container)
        self.assertEqual(headers["Authorization"], "Bearer sk-sf-synthetic")
        with patch.dict(os.environ, {"RERANK_BASE_URL": "http://reranker.local/v1"}):
            url, _, headers = chat_ranking._rerank_endpoint(container)
        # 自定义重排地址不能收到数据库中的 SiliconFlow Key。
        self.assertEqual(url, "http://reranker.local/v1/rerank")
        self.assertNotIn("Authorization", headers)

        provider = web_search.configured_provider(container)
        self.assertEqual(provider._api_key, "sk-ds-synthetic")

    def test_unreadable_keys_stop_consumers_before_any_upstream_call(self):
        self._save(SILICONFLOW, "sk-sf")
        self._save(web_search.DEEPSEEK_PROVIDER_NAME, "sk-ds",
                   base_url="https://api.deepseek.com/v1")
        self._set_raw_key(SILICONFLOW, CORRUPT_CIPHERTEXT)
        self._set_raw_key(web_search.DEEPSEEK_PROVIDER_NAME, CORRUPT_CIPHERTEXT)
        container = get_application_container()

        with self.assertRaises(ServiceUnavailableError) as raised:
            container.create_query_embeddings()
        self.assertEqual(raised.exception.code, "model_key_unavailable")

        client = SimpleNamespace(post=AsyncMock())
        documents = [SimpleNamespace(page_content="候选", metadata={})]
        with patch.object(chat_ranking, "_get_rerank_client", new=AsyncMock(return_value=client)):
            ranked = asyncio.run(chat_ranking.rerank("问题", documents, container=container))
        client.post.assert_not_called()
        self.assertEqual(ranked, documents)

        with patch.object(web_search.DeepSeekSearchProvider, "search") as search:
            outcome = asyncio.run(web_search.search_web("q", "q", 1, container))
        search.assert_not_called()
        self.assertEqual(outcome.error, "读取 DeepSeek 密钥失败")

    def test_newly_saved_key_reaches_new_clients_immediately(self):
        self._save(SILICONFLOW, "sk-old")
        old = llm_config.get_query_embeddings(app_storage.model_providers)
        self._save(SILICONFLOW, "sk-new")
        new = llm_config.get_query_embeddings(app_storage.model_providers)

        self.assertEqual(old.openai_api_key.get_secret_value(), "sk-old")
        self.assertEqual(new.openai_api_key.get_secret_value(), "sk-new")


class ModelApiTest(IsolatedStorageAsyncTestCase):
    """经 HTTP 验证模型接口的状态码和不回显密钥。"""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        app_storage.initialize_database()
        app_storage.bootstrap_defaults()
        from services.core import config as application_config

        self.client = api_client()
        self.addAsyncCleanup(self.client.aclose)
        login = await self.client.post(
            "/api/login",
            json={
                "username": application_config.INITIAL_ADMIN_USERNAME,
                "password": application_config.initial_admin_password(),
            },
        )
        self.assertEqual(login.status_code, 200)

    async def _save(self, **body):
        return await self.client.post(
            "/api/models",
            json={
                "provider": "模型 A",
                "base_url": "https://example.test/v1",
                "default_model": "m",
                **body,
            },
        )

    async def test_responses_never_contain_plaintext_or_ciphertext(self):
        saved = await self._save(api_key="sk-visible-canary")
        self.assertEqual(saved.status_code, 200)
        stored = app_storage.model_providers.get_config("模型 A").encrypted_api_key
        listed = await self.client.get("/api/models")
        checked = await self.client.get("/api/models/check", params={"provider": "模型 A"})
        for response in (saved, listed, checked):
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("sk-visible-canary", response.text)
            self.assertNotIn(stored, response.text)

    async def test_http_status_codes_follow_the_contract(self):
        missing = await self.client.get("/api/models/check", params={"provider": "不存在"})
        self.assertEqual((missing.status_code, missing.json()["code"]),
                         (404, "model_provider_not_found"))

        conflict = await self._save(api_key="sk-x", clear_api_key=True)
        self.assertEqual((conflict.status_code, conflict.json()["code"]),
                         (422, "invalid_model_configuration"))
        self.assertNotIn("sk-x", conflict.text)

        bad_url = await self._save(api_key="sk-x", base_url="https://u:p@example.test/v1")
        self.assertEqual((bad_url.status_code, bad_url.json()["code"]),
                         (422, "invalid_model_configuration"))

        await self._save(api_key="sk-x")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE models_config SET api_key = ? WHERE provider_name = '模型 A'",
                (CORRUPT_CIPHERTEXT,),
            )
        listed = await self.client.get("/api/models")
        self.assertEqual(listed.status_code, 200)
        broken = await self.client.get("/api/models/check", params={"provider": "模型 A"})
        self.assertEqual((broken.status_code, broken.json()["code"]),
                         (503, "model_key_unavailable"))
        self.assertNotIn(CORRUPT_CIPHERTEXT, broken.text)
        activate = await self.client.put("/api/models/模型 A/active")
        self.assertEqual(activate.status_code, 503)

    async def test_validation_errors_do_not_echo_the_api_key(self):
        canary = "sk-validation-canary-" + "x" * 4096
        response = await self._save(api_key=canary)
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("sk-validation-canary", response.text)


if __name__ == "__main__":
    unittest.main()
