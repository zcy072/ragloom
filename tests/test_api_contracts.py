import sqlite3
import unittest
from unittest.mock import patch

import httpx

import api
from services.chat import sessions as chat_sessions
from services.core import config as application_config
from services.core import model_api_keys, model_use_cases
from services.retrieval.chunk_index import get_chunk_index
from tests.support import IsolatedStorageAsyncTestCase


class ApiContractTest(IsolatedStorageAsyncTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        api.app_storage.initialize_database()
        api.app_storage.bootstrap_defaults()

        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://test",
        )
        login = await self.client.post(
            "/api/login",
            json={
                "username": application_config.INITIAL_ADMIN_USERNAME,
                "password": application_config.initial_admin_password(),
            },
        )
        self.assertEqual(login.status_code, 200)
        self.headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    async def asyncTearDown(self):
        await self.client.aclose()

    def test_openapi_uses_named_request_and_response_models(self):
        openapi = api.app.openapi()

        create_knowledge_base = openapi["paths"]["/api/kbs"]["post"]
        self.assertEqual(
            create_knowledge_base["requestBody"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/KnowledgeBaseNameRequest",
        )
        self.assertEqual(
            create_knowledge_base["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/SuccessResponse",
        )
        self.assertFalse(
            openapi["components"]["schemas"]["KnowledgeBaseNameRequest"][
                "additionalProperties"
            ]
        )

        update_role = openapi["paths"]["/api/roles/{role_id}"]["put"]
        self.assertEqual(
            update_role["requestBody"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/RolePermissionsRequest",
        )
        self.assertEqual(
            openapi["paths"]["/api/users"]["get"]["responses"]["200"]["content"]
            ["application/json"]["schema"]["$ref"],
            "#/components/schemas/UserPageResponse",
        )
        self.assertEqual(
            openapi["paths"]["/api/roles"]["get"]["responses"]["200"]["content"]
            ["application/json"]["schema"]["$ref"],
            "#/components/schemas/RolePageResponse",
        )
        user_properties = openapi["components"]["schemas"]["UserResponse"]["properties"]
        self.assertNotIn("email", user_properties)
        chat_properties = openapi["components"]["schemas"]["ChatRequest"]["properties"]
        self.assertNotIn("username", chat_properties)

    async def test_invalid_and_unknown_request_fields_are_rejected(self):
        duplicate = await self.client.post(
            "/api/kbs",
            headers=self.headers,
            json={"name": "通用知识库"},
        )
        self.assertEqual(duplicate.status_code, 400)
        self.assertEqual(
            duplicate.json()["code"], "knowledge_base_already_exists"
        )

        response = await self.client.post(
            "/api/kbs",
            headers=self.headers,
            json={"name": "契约测试", "unexpected": True},
        )
        self.assertEqual(response.status_code, 422)

        response = await self.client.post(
            "/api/chat",
            headers=self.headers,
            json={
                "question": "测试",
                "kb_name": "通用知识库",
                "username": "伪造账号",
            },
        )
        self.assertEqual(response.status_code, 422)

    async def test_application_errors_keep_detail_and_expose_a_stable_code(self):
        response = await self.client.get(
            "/api/chat/sessions/missing-session", headers=self.headers
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json(),
            {
                "detail": "会话不存在或无权访问",
                "code": "chat_session_not_found",
            },
        )

        response = await self.client.put(
            "/api/kbs/通用知识库/chunk_config",
            headers=self.headers,
            json={"chunk_size": 200, "chunk_overlap": 200},
        )
        self.assertEqual(response.status_code, 422)

        api.app_storage.knowledge_bases.create("研发部", 800, 150)
        with patch.object(
            get_chunk_index(),
            "delete_kb",
            side_effect=RuntimeError("index unavailable"),
        ):
            response = await self.client.delete(
                "/api/kbs/研发部", headers=self.headers
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["code"], "knowledge_base_index_cleanup_failed"
        )
        self.assertTrue(
            api.app_storage.knowledge_bases.exists("研发部")
        )

        deleted = await self.client.delete("/api/kbs/通用知识库", headers=self.headers)
        renamed = await self.client.put(
            "/api/kbs/通用知识库", headers=self.headers, json={"name": "公共资料"}
        )
        for response in (deleted, renamed):
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["code"], "common_knowledge_base_protected")
        self.assertTrue(api.app_storage.knowledge_bases.exists("通用知识库"))

        response = await self.client.put(
            "/api/kbs/通用知识库/retrieval_config",
            headers=self.headers,
            json={"query_rewrite_enabled": 1},
        )
        self.assertEqual(response.status_code, 422)

        response = await self.client.put(
            "/api/documents/通用知识库/missing.txt/metadata",
            headers=self.headers,
            json={"effective_date": "22-07-2026"},
        )
        self.assertEqual(response.status_code, 422)

        response = await self.client.put(
            "/api/documents/通用知识库/missing.txt/metadata",
            headers=self.headers,
            json={"tags": [f"标签{i}" for i in range(13)]},
        )
        self.assertEqual(response.status_code, 422)

    async def test_model_configuration_errors_expose_stable_codes(self):
        missing = await self.client.put(
            "/api/models/missing-provider/active", headers=self.headers
        )
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(
            missing.json()["code"], "model_provider_not_found"
        )

        unconfigured = await self.client.put(
            "/api/models/OpenAI/active", headers=self.headers
        )
        self.assertEqual(unconfigured.status_code, 422)
        self.assertEqual(
            unconfigured.json()["code"], "invalid_model_provider"
        )

        with patch.object(
            model_use_cases.model_api_keys,
            "encrypt_api_key",
            side_effect=model_api_keys.ModelApiKeyEncryptionError("key service unavailable"),
        ):
            unavailable = await self.client.post(
                "/api/models",
                headers=self.headers,
                json={
                    "provider": "测试模型",
                    "api_key": "test-key",
                    "base_url": "https://example.test/v1",
                    "default_model": "test-model",
                },
            )
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(
            unavailable.json()["code"], "model_key_encryption_unavailable"
        )

    async def test_identity_administration_errors_expose_stable_codes(self):
        invalid_login = await self.client.post(
            "/api/login", json={"username": "admin", "password": "wrong-password"}
        )
        self.assertEqual(invalid_login.status_code, 401)
        self.assertEqual(invalid_login.json()["code"], "invalid_credentials")

        own_role = await self.client.put(
            "/api/roles/1",
            headers=self.headers,
            json={
                "menu_permissions": ["仪表盘统计"],
                "knowledge_base_permissions": ["全公司"],
            },
        )
        self.assertEqual(own_role.status_code, 403)
        self.assertEqual(own_role.json()["code"], "cannot_edit_own_role")

        protected_role = await self.client.delete("/api/roles/1", headers=self.headers)
        self.assertEqual(protected_role.status_code, 409)
        self.assertEqual(
            protected_role.json()["code"], "role_deletion_conflict"
        )

    async def test_role_contract_uses_canonical_field_names(self):
        deprecated = await self.client.post(
            "/api/roles",
            headers=self.headers,
            json={"name": "旧字段角色", "desc": "不再接受"},
        )
        self.assertEqual(deprecated.status_code, 422)

        response = await self.client.post(
            "/api/roles",
            headers=self.headers,
            json={"name": "审阅者", "description": "只读审阅角色"},
        )
        self.assertEqual(response.status_code, 200)

        roles = (await self.client.get("/api/roles", headers=self.headers)).json()["items"]
        role = next(item for item in roles if item["name"] == "审阅者")
        self.assertEqual(role["description"], "只读审阅角色")

        response = await self.client.put(
            f"/api/roles/{role['id']}",
            headers=self.headers,
            json={
                "menu_permissions": ["知识库"],
                "knowledge_base_permissions": ["通用知识库"],
            },
        )
        self.assertEqual(response.status_code, 200)

        roles = (await self.client.get("/api/roles", headers=self.headers)).json()["items"]
        role = next(item for item in roles if item["name"] == "审阅者")
        self.assertEqual(role["knowledge_base_permissions"], ["通用知识库"])

        models = (await self.client.get("/api/models", headers=self.headers)).json()
        self.assertTrue(models)
        self.assertIs(type(models[0]["is_active"]), bool)

        typed_get_endpoints = (
            "/api/dashboard/stats",
            "/api/kbs",
            "/api/kbs/通用知识库/document_filters",
            "/api/kbs/通用知识库/index_jobs",
            "/api/kbs/通用知识库/chunk_config",
            "/api/kbs/通用知识库/retrieval_config",
            "/api/documents/通用知识库",
            "/api/document_parsers",
            "/api/document_formats",
            "/api/users",
        )
        for endpoint in typed_get_endpoints:
            with self.subTest(endpoint=endpoint):
                response = await self.client.get(endpoint, headers=self.headers)
                self.assertEqual(response.status_code, 200, response.text)

    async def test_identity_list_endpoints_expose_validated_page_metadata(self):
        users = await self.client.get(
            "/api/users?page=99&page_size=10&query=admin", headers=self.headers
        )
        self.assertEqual(users.status_code, 200)
        self.assertEqual(users.json()["page"], 1)
        self.assertEqual(users.json()["total"], 1)
        self.assertEqual(users.json()["total_pages"], 1)
        self.assertEqual(users.json()["items"][0]["username"], "admin")

        roles = await self.client.get(
            "/api/roles?page=99&page_size=10&query=超级", headers=self.headers
        )
        self.assertEqual(roles.status_code, 200)
        self.assertEqual(roles.json()["page"], 1)
        self.assertEqual(roles.json()["total"], 1)
        self.assertEqual(roles.json()["items"][0]["name"], "超级管理员")

        invalid_page_size = await self.client.get(
            "/api/users?page_size=9", headers=self.headers
        )
        self.assertEqual(invalid_page_size.status_code, 422)

    async def test_chat_session_rename_validates_title_and_ownership(self):
        with sqlite3.connect(api.app_storage.DB_FILE) as conn:
            admin_id = conn.execute(
                "SELECT id FROM accounts WHERE username = ?",
                (application_config.INITIAL_ADMIN_USERNAME,),
            ).fetchone()[0]
            conn.executemany(
                "INSERT INTO chat_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    ("mine", admin_id, "admin", "通用知识库", "旧标题", "2026-01-01", "2026-01-01"),
                    ("theirs", admin_id + 1000, "other", "通用知识库", "别人的", "2026-01-01", "2026-01-01"),
                ],
            )

        renamed = await self.client.put(
            "/api/chat/sessions/mine/title",
            headers=self.headers,
            json={"title": "  新标题  "},
        )
        self.assertEqual(renamed.status_code, 200)
        listed = await self.client.get("/api/chat/sessions", headers=self.headers)
        self.assertEqual([item["title"] for item in listed.json()], ["新标题"])

        for invalid in ("", "   ", "长" * 51):
            with self.subTest(title=invalid):
                response = await self.client.put(
                    "/api/chat/sessions/mine/title",
                    headers=self.headers,
                    json={"title": invalid},
                )
                self.assertEqual(response.status_code, 422)
        unknown_field = await self.client.put(
            "/api/chat/sessions/mine/title",
            headers=self.headers,
            json={"title": "标题", "kb_name": "其他库"},
        )
        self.assertEqual(unknown_field.status_code, 422)

        for session_id in ("theirs", "missing"):
            with self.subTest(session_id=session_id):
                response = await self.client.put(
                    f"/api/chat/sessions/{session_id}/title",
                    headers=self.headers,
                    json={"title": "抢改"},
                )
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json()["code"], "chat_session_not_found")
        with sqlite3.connect(api.app_storage.DB_FILE) as conn:
            self.assertEqual(
                conn.execute("SELECT title FROM chat_sessions WHERE id = 'theirs'").fetchone()[0],
                "别人的",
            )

    async def test_renamed_session_title_survives_later_messages(self):
        with sqlite3.connect(api.app_storage.DB_FILE) as conn:
            admin_id = conn.execute(
                "SELECT id FROM accounts WHERE username = ?",
                (application_config.INITIAL_ADMIN_USERNAME,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO chat_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("s1", admin_id, "admin", "通用知识库", "通用知识库 会话", "2026-01-01", "2026-01-01"),
            )
        # 会话还没有消息时改名，随后保存的第一条消息不能把手动标题覆盖回自动标题。
        await self.client.put(
            "/api/chat/sessions/s1/title", headers=self.headers, json={"title": "我的标题"}
        )
        chat_sessions.save_user_message("s1", "第一个问题")

        listed = await self.client.get("/api/chat/sessions", headers=self.headers)
        self.assertEqual(listed.json()[0]["title"], "我的标题")


if __name__ == "__main__":
    unittest.main()
