import asyncio
import base64
import hashlib
import hmac
import json
import sqlite3
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import api
import app_storage
from routers import auth, schemas
from routers import chat as chat_router
from routers import roles as role_router
from routers import users as user_router
from services.core import config as application_config
from services.core.errors import (
    AccessDeniedError,
    AuthenticationError,
    InvalidRequestError,
    RateLimitedError,
)
from services.identity import auth as auth_service
from services.identity import rate_limit
from services.identity import use_cases as identity_use_cases
from tests.support import IsolatedStorageTestCase


class AuthTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        app_storage.bootstrap_defaults()

    def _login_admin(self):
        return auth.login(
            schemas.LoginRequest(
                username=application_config.INITIAL_ADMIN_USERNAME,
                password=application_config.initial_admin_password(),
            )
        )

    def test_unified_login_resolves_super_administrator_permissions(self):
        result = self._login_admin()
        principal = auth_service.current_principal(
            HTTPAuthorizationCredentials(
                scheme="Bearer", credentials=result["access_token"]
            )
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(principal.role, "超级管理员")
        self.assertIn("用户管理", principal.menu_permissions)
        self.assertEqual(principal.knowledge_base_permissions, ("全公司",))

    def test_password_hash_is_salted_and_database_value_cannot_be_replayed(self):
        first = application_config.hash_password("same-password")
        second = application_config.hash_password("same-password")
        self.assertNotEqual(first, second)
        self.assertTrue(application_config.verify_password("same-password", first))

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            stored = conn.execute(
                "SELECT password FROM accounts WHERE username = 'admin'"
            ).fetchone()[0]
        with self.assertRaises(AuthenticationError) as error:
            auth.login(schemas.LoginRequest(username="admin", password=stored))
        self.assertEqual(error.exception.code, "invalid_credentials")

    def test_tampered_access_token_is_rejected(self):
        token = self._login_admin()["access_token"]
        tampered = f"{token[:-1]}{'a' if token[-1] != 'a' else 'b'}"
        with self.assertRaises(HTTPException) as error:
            auth_service.current_principal(
                HTTPAuthorizationCredentials(scheme="Bearer", credentials=tampered)
            )
        self.assertEqual(error.exception.status_code, 401)

    def test_signed_non_object_access_token_is_rejected(self):
        body = base64.urlsafe_b64encode(b"[]").rstrip(b"=").decode("ascii")
        signature = hmac.new(
            auth_service._signing_key(), body.encode("ascii"), hashlib.sha256
        ).digest()
        encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
        with self.assertRaises(HTTPException) as error:
            auth_service.current_principal(
                HTTPAuthorizationCredentials(
                    scheme="Bearer", credentials=f"{body}.{encoded_signature}"
                )
            )
        self.assertEqual(error.exception.status_code, 401)

    def test_concurrent_first_requests_agree_on_one_generated_signing_key(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute("DELETE FROM app_settings WHERE key = 'auth_signing_key'")
        auth_service._signing_key_cache.clear()
        start = threading.Barrier(8)
        keys: list[bytes] = []
        errors: list[Exception] = []

        def first_request():
            try:
                start.wait()
                keys.append(auth_service._signing_key())
            except Exception as exc:  # 竞争失败（IntegrityError 等）会落在这里
                errors.append(exc)

        with patch.dict("os.environ", {"AUTH_SECRET_KEY": ""}):
            threads = [threading.Thread(target=first_request) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(set(keys)), 1)
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            (stored,) = conn.execute(
                "SELECT value FROM app_settings WHERE key = 'auth_signing_key'"
            ).fetchone()
        self.assertEqual(keys[0], stored.encode("utf-8"))

    def test_new_access_tokens_read_the_current_ttl_setting(self):
        with patch.object(
            auth_service,
            "get_settings",
            return_value=SimpleNamespace(
                auth_secret_key="test-signing-key", auth_token_ttl_seconds=120
            ),
        ):
            token = auth_service.issue_access_token(1, "admin", "credential")

        payload = json.loads(auth_service._urlsafe_decode(token.split(".", 1)[0]))
        self.assertEqual(payload["exp"] - payload["iat"], 120)

    def test_http_endpoints_use_one_login_route(self):
        async def exercise_api():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                missing = await client.post(
                    "/api/login/admin", json={"username": "admin", "password": "x"}
                )
                login = await client.post(
                    "/api/login",
                    json={
                        "username": "admin",
                        "password": application_config.initial_admin_password(),
                    },
                )
                kbs = await client.get(
                    "/api/kbs",
                    headers={"Authorization": f"Bearer {login.json()['access_token']}"},
                )
                return missing, login, kbs

        missing, login, kbs = asyncio.run(exercise_api())
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(login.status_code, 200)
        self.assertEqual(kbs.status_code, 200)

    def test_account_permissions_filter_knowledge_bases(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.executemany(
                "INSERT INTO knowledge_bases (name, kb_id, created_at) VALUES (?, lower(hex(randomblob(16))), ?)",
                [("研发知识库", application_config.db_now()), ("财务知识库", application_config.db_now())],
            )
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, knowledge_base_permissions, created_at) VALUES ('研发成员', '', '[\"问答对话\"]', '[\"研发知识库\"]', ?)",
                (application_config.db_now(),),
            )
            role_id = conn.execute("SELECT id FROM roles WHERE name = '研发成员'").fetchone()[0]
            conn.execute(
                "INSERT INTO accounts (name, phone, status, role_id, created_at, password, username) VALUES ('研发用户', '', '正常', ?, ?, ?, 'developer')",
                (role_id, application_config.db_now(), application_config.hash_password("user-password")),
            )

        async def exercise_permissions():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                login = await client.post(
                    "/api/login", json={"username": "developer", "password": "user-password"}
                )
                headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
                return (
                    await client.get("/api/kbs", headers=headers),
                    await client.get("/api/documents/研发知识库", headers=headers),
                    await client.get("/api/documents/财务知识库", headers=headers),
                    await client.get("/api/users", headers=headers),
                )

        listing, allowed, forbidden, users = asyncio.run(exercise_permissions())
        self.assertEqual([item["name"] for item in listing.json()], ["研发知识库"])
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(users.status_code, 403)

    def test_me_returns_live_permissions_instead_of_login_time_snapshot(self):
        login = self._login_admin()
        headers = {"Authorization": f"Bearer {login['access_token']}"}

        async def fetch_me():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/api/me", headers=headers), await client.get("/api/me")

        me, anonymous = asyncio.run(fetch_me())
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["username"], "admin")
        self.assertEqual(me.json()["menuPermissions"], login["user"]["menuPermissions"])
        self.assertEqual(anonymous.status_code, 401)

    def test_short_password_is_rejected_on_register_and_change(self):
        with self.assertRaises(InvalidRequestError) as registered:
            auth.register_user(
                schemas.RegistrationRequest(
                    username="shortpw", name="短密码", phone="", password="1234567"
                )
            )
        self.assertIn("至少", str(registered.exception))

        login = self._login_admin()
        principal = auth_service.current_principal(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=login["access_token"])
        )
        with self.assertRaises(InvalidRequestError):
            auth.change_password(
                schemas.PasswordChangeRequest(
                    old_password=application_config.initial_admin_password(),
                    new_password="1234567",
                ),
                principal,
            )
        # 被拒绝的改密不能让原密码失效。
        self.assertEqual(
            auth.login(
                schemas.LoginRequest(
                    username="admin", password=application_config.initial_admin_password()
                )
            )["status"],
            "success",
        )

    def test_registration_rejects_malformed_username_and_phone(self):
        for username, phone in (("带 空格", ""), ("tab\tname", ""), ("good", "不是号码")):
            with self.subTest(username=username, phone=phone):
                with self.assertRaises(InvalidRequestError):
                    auth.register_user(
                        schemas.RegistrationRequest(
                            username=username, name="姓名", phone=phone, password="password-1"
                        )
                    )
        auth.register_user(
            schemas.RegistrationRequest(
                username="张三.li@dev-1", name="张三", phone="+86 138-0000-0000",
                password="password-1",
            )
        )

    def test_password_over_bcrypt_limit_is_a_client_error_not_a_server_crash(self):
        # bcrypt 只接受 72 字节；30 个汉字是 90 字节。注册和改密都应返回可读的 422，而不是 500。
        too_long = "密" * 30
        with self.assertRaises(InvalidRequestError) as registered:
            auth.register_user(
                schemas.RegistrationRequest(
                    username="longpw", name="长密码", phone="13800000000", password=too_long
                )
            )
        self.assertIn("72", str(registered.exception))

        login = self._login_admin()
        principal = auth_service.current_principal(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=login["access_token"])
        )
        with self.assertRaises(InvalidRequestError):
            auth.change_password(
                schemas.PasswordChangeRequest(
                    old_password=application_config.initial_admin_password(),
                    new_password=too_long,
                ),
                principal,
            )
        # 失败的改密不能让原密码失效。
        self.assertEqual(
            auth.login(
                schemas.LoginRequest(
                    username="admin", password=application_config.initial_admin_password()
                )
            )["status"],
            "success",
        )

    def test_password_change_invalidates_existing_access_token(self):
        login = self._login_admin()
        principal = auth_service.current_principal(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=login["access_token"])
        )
        auth.change_password(
            schemas.PasswordChangeRequest(
                old_password=application_config.initial_admin_password(),
                new_password="replacement-password",
            ),
            principal,
        )
        with self.assertRaises(HTTPException) as error:
            auth_service.current_principal(
                HTTPAuthorizationCredentials(scheme="Bearer", credentials=login["access_token"])
            )
        self.assertEqual(error.exception.status_code, 401)
        self.assertEqual(
            auth.login(schemas.LoginRequest(username="admin", password="replacement-password"))["status"],
            "success",
        )

    def test_account_cannot_change_its_own_role_status_or_delete_itself(self):
        principal = auth_service.current_principal(
            HTTPAuthorizationCredentials(
                scheme="Bearer", credentials=self._login_admin()["access_token"]
            )
        )
        for operation in (
            lambda: user_router.update_user_role(
                principal.account_id, schemas.UserRoleRequest(role_id=principal.role_id), principal
            ),
            lambda: user_router.update_user_status(
                principal.account_id, schemas.UserStatusRequest(status="停用"), principal
            ),
            lambda: user_router.delete_user(principal.account_id, principal),
        ):
            with self.assertRaises(AccessDeniedError) as error:
                operation()
            self.assertIn("不能", error.exception.public_message)

    def test_account_cannot_edit_its_current_role(self):
        principal = auth_service.current_principal(
            HTTPAuthorizationCredentials(
                scheme="Bearer", credentials=self._login_admin()["access_token"]
            )
        )
        assert principal.role_id is not None
        with self.assertRaises(AccessDeniedError) as error:
            role_router.update_role(
                principal.role_id,
                schemas.RolePermissionsRequest(
                    menu_permissions=list(principal.menu_permissions),
                    knowledge_base_permissions=list(principal.knowledge_base_permissions),
                ),
                principal,
            )
        self.assertEqual(error.exception.code, "cannot_edit_own_role")

    def test_limited_manager_cannot_grant_permissions_it_does_not_have(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, knowledge_base_permissions, created_at) VALUES ('局部管理员', '', '[\"用户管理\", \"角色管理\"]', '[]', ?)",
                (application_config.db_now(),),
            )
            limited_role_id = conn.execute(
                "SELECT id FROM roles WHERE name = '局部管理员'"
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, knowledge_base_permissions, created_at) VALUES ('目标角色', '', '[]', '[]', ?)",
                (application_config.db_now(),),
            )
            target_role_id = conn.execute(
                "SELECT id FROM roles WHERE name = '目标角色'"
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO accounts (username, name, password, status, role_id, created_at) VALUES ('manager', '局部管理员', ?, '正常', ?, ?)",
                (application_config.hash_password("password"), limited_role_id, application_config.db_now()),
            )
        manager = auth_service.current_principal(
            HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=auth.login(
                    schemas.LoginRequest(username="manager", password="password")
                )["access_token"],
            )
        )
        with self.assertRaises(AccessDeniedError) as error:
            role_router.update_role(
                target_role_id,
                schemas.RolePermissionsRequest(menu_permissions=["模型配置"]),
                manager,
            )
        self.assertEqual(error.exception.code, "permission_delegation_denied")


    def _limited_role_manager(self):
        """建一个只有用户管理和角色管理菜单、没有知识库权限的账号，返回其主体。"""
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, knowledge_base_permissions, created_at) VALUES ('局部管理员', '', '[\"用户管理\", \"角色管理\"]', '[]', ?)",
                (application_config.db_now(),),
            )
            role_id = conn.execute("SELECT id FROM roles WHERE name = '局部管理员'").fetchone()[0]
            conn.execute(
                "INSERT INTO accounts (username, name, password, status, role_id, created_at) VALUES ('manager', '局部管理员', ?, '正常', ?, ?)",
                (application_config.hash_password("password"), role_id, application_config.db_now()),
            )
        return auth_service.current_principal(
            HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=auth.login(
                    schemas.LoginRequest(username="manager", password="password")
                )["access_token"],
            )
        )

    def test_limited_manager_cannot_shrink_a_role_with_more_permissions(self):
        manager = self._limited_role_manager()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, knowledge_base_permissions, created_at) VALUES ('平级角色', '', '[\"用户管理\", \"模型配置\"]', '[]', ?)",
                (application_config.db_now(),),
            )
            peer_role_id = conn.execute("SELECT id FROM roles WHERE name = '平级角色'").fetchone()[0]
        with self.assertRaises(AccessDeniedError) as error:
            role_router.update_role(
                peer_role_id, schemas.RolePermissionsRequest(menu_permissions=[]), manager
            )
        self.assertEqual(error.exception.code, "target_permissions_exceed_actor")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            menus = conn.execute(
                "SELECT menu_permissions FROM roles WHERE id = ?", (peer_role_id,)
            ).fetchone()[0]
        self.assertIn("模型配置", menus)

    def test_super_administrator_role_permissions_cannot_be_edited(self):
        manager = self._limited_role_manager()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            super_role_id = conn.execute(
                "SELECT id FROM roles WHERE name = '超级管理员'"
            ).fetchone()[0]
        with self.assertRaises(AccessDeniedError) as error:
            role_router.update_role(
                super_role_id, schemas.RolePermissionsRequest(menu_permissions=[]), manager
            )
        self.assertEqual(error.exception.code, "protected_role")

    def test_limited_manager_can_edit_a_role_within_its_scope(self):
        manager = self._limited_role_manager()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, knowledge_base_permissions, created_at) VALUES ('下级角色', '', '[\"用户管理\"]', '[]', ?)",
                (application_config.db_now(),),
            )
            role_id = conn.execute("SELECT id FROM roles WHERE name = '下级角色'").fetchone()[0]
        role_router.update_role(
            role_id, schemas.RolePermissionsRequest(menu_permissions=["角色管理"]), manager
        )
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            menus = conn.execute(
                "SELECT menu_permissions FROM roles WHERE id = ?", (role_id,)
            ).fetchone()[0]
        self.assertEqual(menus, '["角色管理"]')

    def test_unknown_username_still_runs_one_bcrypt_check(self):
        with patch.object(
            application_config.bcrypt, "checkpw", wraps=application_config.bcrypt.checkpw
        ) as checkpw:
            with self.assertRaises(AuthenticationError):
                identity_use_cases.authenticate_account(
                    db_file=app_storage.DB_FILE, username="nobody", password="password"
                )
        self.assertEqual(checkpw.call_count, 1)


    def _limited_user_manager(self):
        """建一个只有用户管理权限的账号，返回其主体。"""
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, knowledge_base_permissions, created_at) VALUES ('用户管理员', '', '[\"用户管理\"]', '[]', ?)",
                (application_config.db_now(),),
            )
            role_id = conn.execute("SELECT id FROM roles WHERE name = '用户管理员'").fetchone()[0]
            conn.execute(
                "INSERT INTO accounts (username, name, password, status, role_id, created_at) VALUES ('usermgr', '用户管理员', ?, '正常', ?, ?)",
                (application_config.hash_password("password"), role_id, application_config.db_now()),
            )
            conn.execute(
                "INSERT INTO accounts (username, name, password, status, created_at) VALUES ('plain', '普通用户', ?, '正常', ?)",
                (application_config.hash_password("password"), application_config.db_now()),
            )
            plain_id = conn.execute("SELECT id FROM accounts WHERE username = 'plain'").fetchone()[0]
        manager = auth_service.current_principal(
            HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=auth.login(
                    schemas.LoginRequest(username="usermgr", password="password")
                )["access_token"],
            )
        )
        return manager, plain_id

    def test_limited_manager_cannot_manage_an_account_with_more_permissions(self):
        manager, _plain_id = self._limited_user_manager()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            admin_id, admin_role_id = conn.execute(
                "SELECT id, role_id FROM accounts WHERE username = ?",
                (application_config.INITIAL_ADMIN_USERNAME,),
            ).fetchone()
        for operation in (
            lambda: user_router.delete_user(admin_id, manager),
            lambda: user_router.update_user_status(
                admin_id, schemas.UserStatusRequest(status="停用"), manager
            ),
            lambda: user_router.update_user_role(
                admin_id, schemas.UserRoleRequest(role_id=admin_role_id), manager
            ),
        ):
            with self.assertRaises(AccessDeniedError) as error:
                operation()
            self.assertEqual(error.exception.code, "target_permissions_exceed_actor")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            self.assertEqual(
                conn.execute("SELECT status FROM accounts WHERE id = ?", (admin_id,)).fetchone()[0],
                "正常",
            )

    def test_limited_manager_can_still_manage_an_account_within_its_scope(self):
        manager, plain_id = self._limited_user_manager()
        user_router.update_user_status(plain_id, schemas.UserStatusRequest(status="停用"), manager)
        user_router.delete_user(plain_id, manager)
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            self.assertIsNone(
                conn.execute("SELECT 1 FROM accounts WHERE id = ?", (plain_id,)).fetchone()
            )

    def test_registration_username_race_reports_a_taken_name_not_a_server_error(self):
        from services.core.errors import InvalidStateError
        from services.identity import accounts, use_cases

        with patch.object(
            accounts, "register_user", side_effect=sqlite3.IntegrityError("UNIQUE constraint failed")
        ), self.assertRaises(InvalidStateError) as error:
            use_cases.register_account(
                db_file=app_storage.DB_FILE,
                now=application_config.db_now,
                username="racer",
                name="并发",
                phone="",
                password="racer-pass-1",
            )
        self.assertEqual(error.exception.code, "username_already_registered")

    def test_unexpected_registration_error_does_not_leak_internal_details(self):
        from services.core.errors import OperationFailedError
        from services.identity import accounts, use_cases

        with patch.object(
            accounts, "register_user", side_effect=sqlite3.OperationalError("disk I/O error at /app/storage")
        ), self.assertLogs("services.identity.use_cases", "ERROR"), self.assertRaises(
            OperationFailedError
        ) as error:
            use_cases.register_account(
                db_file=app_storage.DB_FILE,
                now=application_config.db_now,
                username="broken",
                name="异常",
                phone="",
                password="broken-pass-1",
            )
        self.assertNotIn("/app/storage", error.exception.public_message)

    def test_duplicate_role_name_reports_a_clean_conflict(self):
        from services.core.errors import InvalidStateError
        from services.identity import use_cases

        with self.assertRaises(InvalidStateError) as error:
            use_cases.create_role(
                db_file=app_storage.DB_FILE,
                now=application_config.db_now,
                name="超级管理员",
                description="",
            )
        self.assertEqual(error.exception.code, "role_already_exists")
        self.assertNotIn("UNIQUE", error.exception.public_message)

    def test_login_and_registration_reject_blank_or_oversized_fields(self):
        from pydantic import ValidationError

        for build in (
            lambda: schemas.LoginRequest(username="  ", password="x"),
            lambda: schemas.LoginRequest(username="a" * 81, password="x"),
            lambda: schemas.RegistrationRequest(
                username="", name="名字", phone="", password="password-1"
            ),
            lambda: schemas.RegistrationRequest(
                username="u", name="名字", phone="1" * 33, password="password-1"
            ),
            lambda: schemas.RegistrationRequest(
                username="u", name="名字", phone="", password="p" * 257
            ),
        ):
            with self.assertRaises(ValidationError):
                build()

    def test_malformed_permission_columns_degrade_to_no_permissions(self):
        from services.identity.permissions import decode_permission_list

        with self.assertLogs("services.identity.permissions", "WARNING"):
            self.assertEqual(decode_permission_list("not-json"), [])
        with self.assertLogs("services.identity.permissions", "WARNING"):
            self.assertEqual(decode_permission_list('{"a": 1}'), [])
        self.assertEqual(decode_permission_list('["用户管理", 3]'), ["用户管理"])
        self.assertEqual(decode_permission_list(None), [])

class EventWindowTest(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0

    def _window(self, max_events=3, window=60, max_keys=10_000):
        return rate_limit.EventWindow(
            max_events, window, max_keys=max_keys, clock=lambda: self.now
        )

    def test_key_is_blocked_only_after_reaching_the_limit(self):
        window = self._window()
        for _ in range(2):
            window.record("alice")
        self.assertEqual(window.retry_after("alice"), 0)

        window.record("alice")

        self.assertEqual(window.retry_after("alice"), 60)
        self.assertEqual(window.retry_after("bob"), 0)

    def test_block_expires_when_the_oldest_event_leaves_the_window(self):
        window = self._window()
        for offset in (0, 10, 20):
            self.now = 1000.0 + offset
            window.record("alice")

        self.now = 1030.0
        self.assertEqual(window.retry_after("alice"), 30)
        self.now = 1060.0
        self.assertEqual(window.retry_after("alice"), 0)

    def test_recording_while_blocked_does_not_extend_the_block(self):
        window = self._window()
        for _ in range(3):
            window.record("alice")
        self.now = 1030.0
        for _ in range(10):
            window.record("alice")

        self.assertEqual(window.retry_after("alice"), 30)

    def test_zero_limit_or_window_disables_the_check(self):
        for window in (self._window(max_events=0), self._window(window=0)):
            for _ in range(100):
                window.record("alice")
            self.assertEqual(window.retry_after("alice"), 0)

    def test_tracked_keys_are_bounded_when_many_distinct_keys_are_submitted(self):
        window = self._window(max_keys=100)
        for index in range(1000):
            window.record(f"user-{index}")

        self.assertLessEqual(len(window._events), 100)

    def test_expired_keys_are_dropped_before_live_ones(self):
        window = self._window(max_keys=3)
        window.record("old-1")
        window.record("old-2")
        self.now += 120
        window.record("live")
        window.record("new")

        self.assertIn("live", window._events)
        self.assertNotIn("old-1", window._events)


class LoginRateLimitTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        app_storage.bootstrap_defaults()
        self.now = 1000.0
        self.limiter = rate_limit.AuthRateLimiter(
            login_failures=3,
            login_window_seconds=600,
            client_failures=6,
            client_window_seconds=300,
            register_attempts=2,
            register_window_seconds=3600,
            clock=lambda: self.now,
        )
        patcher = patch.object(rate_limit, "_limiter", self.limiter)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _login(self, password, username="admin", client="10.0.0.1"):
        return auth.login(
            schemas.LoginRequest(username=username, password=password), client
        )

    def _admin_password(self):
        return application_config.initial_admin_password()

    def test_repeated_wrong_passwords_lock_the_username_even_for_the_right_password(self):
        for _ in range(3):
            with self.assertRaises(AuthenticationError):
                self._login("wrong")

        with self.assertRaises(RateLimitedError) as locked:
            self._login(self._admin_password())

        self.assertEqual(locked.exception.code, "login_rate_limited")
        self.assertEqual(locked.exception.retry_after_seconds, 600)

    def test_lock_applies_across_client_addresses(self):
        for index in range(3):
            with self.assertRaises(AuthenticationError):
                self._login("wrong", client=f"10.0.0.{index}")

        with self.assertRaises(RateLimitedError):
            self._login(self._admin_password(), client="192.168.1.9")

    def test_lock_expires_after_the_window(self):
        for _ in range(3):
            with self.assertRaises(AuthenticationError):
                self._login("wrong")
        self.now += 601

        self.assertEqual(self._login(self._admin_password())["status"], "success")

    def test_successful_login_resets_the_username_failure_count(self):
        for _ in range(2):
            with self.assertRaises(AuthenticationError):
                self._login("wrong")
        self._login(self._admin_password())
        for _ in range(2):
            with self.assertRaises(AuthenticationError):
                self._login("wrong")

        self.assertEqual(self._login(self._admin_password())["status"], "success")

    def test_locked_requests_are_not_verified_and_do_not_extend_the_lock(self):
        for _ in range(3):
            with self.assertRaises(AuthenticationError):
                self._login("wrong")
        self.now += 300
        with patch.object(
            application_config, "verify_password"
        ) as verify, self.assertRaises(RateLimitedError) as locked:
            self._login("wrong")

        verify.assert_not_called()
        self.assertEqual(locked.exception.retry_after_seconds, 300)

    def test_username_matching_ignores_case_and_surrounding_spaces(self):
        for username in ("admin", "ADMIN", " Admin "):
            with self.assertRaises(AuthenticationError):
                self._login("wrong", username=username)

        with self.assertRaises(RateLimitedError):
            self._login("wrong", username="admin")

    def test_unknown_usernames_are_limited_like_real_ones(self):
        for _ in range(3):
            with self.assertRaises(AuthenticationError):
                self._login("wrong", username="nobody")

        with self.assertRaises(RateLimitedError):
            self._login("wrong", username="nobody")

    def test_one_client_trying_many_usernames_is_limited_by_address(self):
        for index in range(6):
            with self.assertRaises(AuthenticationError):
                self._login("wrong", username=f"user-{index}")

        with self.assertRaises(RateLimitedError):
            self._login("wrong", username="user-new")
        # 其他来源地址不受影响。
        with self.assertRaises(AuthenticationError):
            self._login("wrong", username="user-new", client="10.0.0.2")

    def test_inactive_account_is_not_counted_as_a_failed_login(self):
        auth.register_user(
            schemas.RegistrationRequest(
                username="pending", name="待审核", phone="", password="pending-pass-1"
            ),
            "10.0.0.5",
        )
        for _ in range(5):
            with self.assertRaises(AccessDeniedError):
                self._login("pending-pass-1", username="pending")

    def test_registration_attempts_are_limited_per_client_address(self):
        for index in range(2):
            auth.register_user(
                schemas.RegistrationRequest(
                    username=f"new-{index}", name="新用户", phone="", password="new-pass-123"
                ),
                "10.0.0.7",
            )

        with self.assertRaises(RateLimitedError) as locked:
            auth.register_user(
                schemas.RegistrationRequest(
                    username="new-3", name="新用户", phone="", password="new-pass-123"
                ),
                "10.0.0.7",
            )
        self.assertEqual(locked.exception.code, "registration_rate_limited")
        auth.register_user(
            schemas.RegistrationRequest(
                username="other", name="其他", phone="", password="new-pass-123"
            ),
            "10.0.0.8",
        )

    def test_empty_client_address_skips_address_based_limits(self):
        for index in range(10):
            with self.assertRaises(AuthenticationError):
                self._login("wrong", username=f"user-{index}", client="")

    def test_http_response_is_429_with_retry_after_header(self):
        async def exercise():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                for _ in range(4):
                    response = await client.post(
                        "/api/login", json={"username": "admin", "password": "wrong"}
                    )
                return response

        response = asyncio.run(exercise())

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "600")
        self.assertEqual(response.json()["code"], "login_rate_limited")
        self.assertIn("请 10 分钟后再试", response.json()["detail"])


class ClientAddressTest(unittest.TestCase):
    @staticmethod
    def _request(headers=None, host="172.18.0.5"):
        return SimpleNamespace(
            headers={k.lower(): v for k, v in (headers or {}).items()},
            client=SimpleNamespace(host=host) if host else None,
        )

    def test_x_real_ip_is_ignored_unless_trusted(self):
        request = self._request({"X-Real-IP": "203.0.113.9"})
        with patch.dict("os.environ", {"AUTH_TRUST_X_REAL_IP": "false"}):
            self.assertEqual(auth.client_address(request), "172.18.0.5")

    def test_x_real_ip_is_used_when_trusted(self):
        request = self._request({"X-Real-IP": "203.0.113.9"})
        with patch.dict("os.environ", {"AUTH_TRUST_X_REAL_IP": "true"}):
            self.assertEqual(auth.client_address(request), "203.0.113.9")

    def test_falls_back_to_connection_address_without_the_header(self):
        with patch.dict("os.environ", {"AUTH_TRUST_X_REAL_IP": "true"}):
            self.assertEqual(auth.client_address(self._request()), "172.18.0.5")
            self.assertEqual(auth.client_address(self._request(host=None)), "")


if __name__ == "__main__":
    unittest.main()


class ChatRequestLimiterTest(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.limiter = rate_limit.ChatRequestLimiter(
            max_concurrent=2, per_minute=3, stale_seconds=900, clock=lambda: self.now
        )

    def test_concurrent_streams_are_capped_per_account_and_freed_on_release(self):
        first = self.limiter.acquire(1)
        self.limiter.acquire(1)
        with self.assertRaises(RateLimitedError) as error:
            self.limiter.acquire(1)
        self.assertEqual(error.exception.code, "chat_concurrency_limited")
        # 其他账号不受影响。
        self.limiter.acquire(2)()
        first()
        first()  # 重复释放不会多还名额。
        self.limiter.acquire(1)

    def test_requests_per_minute_are_capped_until_the_window_passes(self):
        for _ in range(3):
            self.limiter.acquire(1)()
        with self.assertRaises(RateLimitedError) as error:
            self.limiter.acquire(1)
        self.assertEqual(error.exception.code, "chat_rate_limited")
        self.assertEqual(error.exception.retry_after_seconds, 60)
        self.now += 60
        self.limiter.acquire(1)

    def test_rejected_requests_do_not_use_up_the_minute_quota(self):
        self.limiter.acquire(1)
        self.limiter.acquire(1)
        for _ in range(5):
            with self.assertRaises(RateLimitedError):
                self.limiter.acquire(1)
        self.now += 1000  # 两个未释放的名额已过期，每分钟计数也已清空。
        self.limiter.acquire(1)

    def test_zero_disables_both_limits(self):
        limiter = rate_limit.ChatRequestLimiter(max_concurrent=0, per_minute=0)
        for _ in range(20):
            limiter.acquire(1)


class ChatRateLimitHttpTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        app_storage.bootstrap_defaults()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO knowledge_bases (name, kb_id, created_at) VALUES ('技术部', lower(hex(randomblob(16))), ?)",
                (application_config.db_now(),),
            )

    def _chat(self, count):
        async def events():
            yield "data: {}\n\n"

        async def fake_chat(*_args, **_kwargs):
            return SimpleNamespace(events=events(), refresh_summary=lambda: asyncio.sleep(0))

        async def run():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                login = await client.post(
                    "/api/login",
                    json={
                        "username": application_config.INITIAL_ADMIN_USERNAME,
                        "password": application_config.initial_admin_password(),
                    },
                )
                headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
                return [
                    await client.post(
                        "/api/chat", json={"question": "问题", "kb_name": "技术部"}, headers=headers
                    )
                    for _ in range(count)
                ]

        with patch.dict("os.environ", {"CHAT_USER_MAX_CONCURRENT": "1", "CHAT_USER_PER_MINUTE": "2"}):
            rate_limit.reset_auth_rate_limiter()
            with patch.object(chat_router.chat_service, "chat", side_effect=fake_chat):
                return asyncio.run(run())

    def test_finished_streams_free_the_slot_and_the_minute_cap_returns_429(self):
        responses = self._chat(3)
        # 并发上限是 1：前两次能成功说明流结束后名额已释放。
        self.assertEqual([response.status_code for response in responses], [200, 200, 429])
        self.assertEqual(responses[2].json()["code"], "chat_rate_limited")
        self.assertIn("Retry-After", responses[2].headers)
