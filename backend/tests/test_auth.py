import asyncio
import concurrent.futures
import hashlib
import sqlite3
import threading
import time
import unittest
from http.cookies import SimpleCookie
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import HTTPException, Response
from starlette.requests import Request

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
    ConflictError,
    InvalidRequestError,
    InvalidStateError,
    RateLimitedError,
)
from services.identity import accounts, rate_limit, sessions
from services.identity import auth as auth_service
from services.identity import use_cases as identity_use_cases
from tests.support import WRITE_HEADERS, IsolatedStorageTestCase, add_knowledge_bases, api_client

ADMIN = application_config.INITIAL_ADMIN_USERNAME


def fake_request(token: str | None = None, path: str = "/api/login") -> Request:
    """构造一个可选携带会话 Cookie 的请求对象，供直接调用路由函数的测试使用。"""
    headers = []
    if token is not None:
        headers.append((b"cookie", f"{sessions.COOKIE_NAME}={token}".encode("latin-1")))
    return Request({"type": "http", "method": "POST", "path": path, "headers": headers})


def cookie_token(response) -> str:
    """从响应的 Set-Cookie 中取出会话令牌原值。"""
    jar = SimpleCookie()
    jar.load(response.headers["set-cookie"])
    return jar[sessions.COOKIE_NAME].value


def set_cookie_attributes(header: str) -> dict[str, str]:
    """把一个 Set-Cookie 头拆成小写属性名到值的映射（无值属性映射为空串）。"""
    attributes = {}
    for part in header.split(";")[1:]:
        name, _, value = part.strip().partition("=")
        attributes[name.lower()] = value
    return attributes


def run(coroutine_factory):
    return asyncio.run(coroutine_factory())


class AuthTestBase(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        app_storage.bootstrap_defaults()

    def _login(self, username=ADMIN, password=None, token=None):
        """直接调用登录路由，返回 (响应 JSON, 新会话令牌)。"""
        response = Response()
        result = auth.login(
            schemas.LoginRequest(
                username=username,
                password=password or application_config.initial_admin_password(),
            ),
            fake_request(token),
            response,
        )
        return result, cookie_token(response)

    def _login_admin(self):
        return self._login()[0]

    def _principal(self, username, password):
        _result, token = self._login(username, password)
        return auth_service.current_principal(fake_request(token, "/api/me"))

    def _admin_principal(self):
        return self._principal(ADMIN, application_config.initial_admin_password())

    def _add_user(self, username, password="user-password", status="正常", role_id=None):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            cursor = conn.execute(
                "INSERT INTO accounts (username, name, password, status, role_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    username,
                    f"{username} 姓名",
                    application_config.hash_password(password),
                    status,
                    role_id,
                    application_config.db_now(),
                ),
            )
            return int(cursor.lastrowid or 0)

    def _session_rows(self, account_id=None):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            if account_id is None:
                return conn.execute("SELECT token_hash, account_id FROM auth_sessions").fetchall()
            return conn.execute(
                "SELECT token_hash, account_id FROM auth_sessions WHERE account_id = ?",
                (account_id,),
            ).fetchall()

    def _stored_password(self, username=ADMIN):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            return conn.execute(
                "SELECT password FROM accounts WHERE username = ?", (username,)
            ).fetchone()[0]

    def _admin_id(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            return conn.execute("SELECT id FROM accounts WHERE username = ?", (ADMIN,)).fetchone()[0]

    @staticmethod
    async def _http_login(client, username=ADMIN, password=None):
        return await client.post(
            "/api/login",
            json={
                "username": username,
                "password": password or application_config.initial_admin_password(),
            },
        )


class AuthTest(AuthTestBase):
    def test_unified_login_resolves_super_administrator_permissions(self):
        result = self._login_admin()
        principal = self._admin_principal()

        self.assertEqual(result["status"], "success")
        self.assertNotIn("access_token", result)
        self.assertNotIn("token_type", result)
        self.assertEqual(principal.role, "超级管理员")
        self.assertIn("用户管理", principal.menu_permissions)
        self.assertTrue(principal.all_knowledge_bases)

    def test_password_hash_is_salted_and_database_value_cannot_be_replayed(self):
        first = application_config.hash_password("same-password")
        second = application_config.hash_password("same-password")
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("$2b$12$"))
        self.assertTrue(application_config.verify_password("same-password", first))

        with self.assertRaises(AuthenticationError) as error:
            self._login(ADMIN, self._stored_password())
        self.assertEqual(error.exception.code, "invalid_credentials")

    def test_http_endpoints_use_one_login_route(self):
        async def exercise():
            async with api_client() as client:
                missing = await client.post(
                    "/api/login/admin", json={"username": "admin", "password": "x"}
                )
                login = await self._http_login(client)
                return missing, login, await client.get("/api/kbs")

        missing, login, kbs = run(exercise)
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(login.status_code, 200)
        self.assertEqual(kbs.status_code, 200)

    def test_account_permissions_filter_knowledge_bases(self):
        dev_id, finance_id = add_knowledge_bases("研发知识库", "财务知识库")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, all_knowledge_bases, created_at) VALUES ('研发成员', '', '[\"问答对话\"]', 0, ?)",
                (application_config.db_now(),),
            )
            role_id = conn.execute("SELECT id FROM roles WHERE name = '研发成员'").fetchone()[0]
            conn.execute(
                "INSERT INTO role_knowledge_bases (role_id, kb_id) VALUES (?, ?)",
                (role_id, dev_id),
            )
        self._add_user("developer", role_id=role_id)

        async def exercise():
            async with api_client() as client:
                await self._http_login(client, "developer", "user-password")
                return (
                    await client.get("/api/kbs"),
                    await client.get(f"/api/documents/{dev_id}"),
                    await client.get(f"/api/documents/{finance_id}"),
                    await client.get("/api/users"),
                )

        listing, allowed, forbidden, users = run(exercise)
        self.assertEqual([item["name"] for item in listing.json()], ["研发知识库"])
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(users.status_code, 403)

    def test_me_returns_live_permissions_and_the_same_projection_as_login(self):
        async def exercise():
            async with api_client() as client:
                login = await self._http_login(client)
                me = await client.get("/api/me")
            async with api_client() as anonymous:
                return login, me, await anonymous.get("/api/me")

        login, me, anonymous = run(exercise)
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json(), login.json()["user"])
        self.assertEqual(me.json()["id"], self._admin_id())
        self.assertEqual(me.json()["username"], ADMIN)
        self.assertIn("name", me.json())
        self.assertEqual(
            set(me.json()),
            {"id", "username", "name", "role", "menuPermissions", "allKnowledgeBases", "knowledgeBaseIds"},
        )
        self.assertEqual(me.headers["cache-control"], "no-store")
        self.assertEqual(login.headers["cache-control"], "no-store")
        self.assertEqual(anonymous.status_code, 401)

    def test_role_change_is_visible_on_the_next_request_without_new_login(self):
        user_id = self._add_user("member")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, all_knowledge_bases, created_at) VALUES ('问答', '', '[\"问答对话\"]', 0, ?)",
                (application_config.db_now(),),
            )

        async def exercise():
            async with api_client() as client:
                await self._http_login(client, "member", "user-password")
                before = await client.get("/api/me")
                with sqlite3.connect(app_storage.DB_FILE) as conn:
                    conn.execute(
                        "UPDATE accounts SET role_id = (SELECT id FROM roles WHERE name = '问答') WHERE id = ?",
                        (user_id,),
                    )
                return before, await client.get("/api/me")

        before, after = run(exercise)
        self.assertEqual((before.json()["role"], before.json()["menuPermissions"]), ("未分配", []))
        self.assertEqual((after.json()["role"], after.json()["menuPermissions"]), ("问答", ["问答对话"]))

    def test_short_password_is_rejected_on_register_and_change(self):
        with self.assertRaises(InvalidRequestError) as registered:
            auth.register_user(
                schemas.RegistrationRequest(
                    username="shortpw", name="短密码", phone="", password="1234567"
                ),
                Response(),
            )
        self.assertIn("至少", str(registered.exception))

        _result, token = self._login()
        with self.assertRaises(InvalidRequestError):
            auth.change_password(
                schemas.PasswordChangeRequest(
                    old_password=application_config.initial_admin_password(),
                    new_password="1234567",
                ),
                fake_request(token),
                Response(),
                auth_service.current_principal(fake_request(token)),
            )
        # 被拒绝的改密不能让原密码失效，也不撤销会话。
        self.assertEqual(self._login_admin()["status"], "success")
        auth_service.current_principal(fake_request(token))

    def test_registration_rejects_malformed_username_and_phone(self):
        for username, phone in (("带 空格", ""), ("tab\tname", ""), ("good", "不是号码")):
            with self.subTest(username=username, phone=phone):
                with self.assertRaises(InvalidRequestError):
                    auth.register_user(
                        schemas.RegistrationRequest(
                            username=username, name="姓名", phone=phone, password="password-1"
                        ),
                        Response(),
                    )
        auth.register_user(
            schemas.RegistrationRequest(
                username="张三.li@dev-1", name="张三", phone="+86 138-0000-0000",
                password="password-1",
            ),
            Response(),
        )

    def test_password_byte_limit_is_72_utf8_bytes(self):
        accepted = {"ascii72": "a" * 72, "cjk72": "密" * 24}
        rejected = {"ascii73": "a" * 73, "cjk75": "密" * 25}
        for username, password in accepted.items():
            auth.register_user(
                schemas.RegistrationRequest(username=username, name="n", phone="", password=password),
                Response(),
            )
        for username, password in rejected.items():
            with self.subTest(username=username), self.assertRaises(InvalidRequestError) as error:
                auth.register_user(
                    schemas.RegistrationRequest(
                        username=username, name="n", phone="", password=password
                    ),
                    Response(),
                )
            self.assertIn("72", str(error.exception))
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute("UPDATE accounts SET status = '正常' WHERE username IN ('ascii72', 'cjk72')")
        self.assertEqual(self._login("ascii72", "a" * 72)[0]["status"], "success")
        self.assertEqual(self._login("cjk72", "密" * 24)[0]["status"], "success")
        # 超长的登录密码按凭证错误处理，不能抛出 500。
        with self.assertRaises(AuthenticationError):
            self._login("ascii72", "a" * 73)

        _result, token = self._login()
        with self.assertRaises(InvalidRequestError):
            auth.change_password(
                schemas.PasswordChangeRequest(
                    old_password=application_config.initial_admin_password(),
                    new_password="密" * 30,
                ),
                fake_request(token),
                Response(),
                auth_service.current_principal(fake_request(token)),
            )
        self.assertEqual(self._login_admin()["status"], "success")

    def test_password_whitespace_is_preserved_while_username_is_trimmed(self):
        password = "  spaced pass  "
        auth.register_user(
            schemas.RegistrationRequest(username="  spacey ", name=" 名 ", phone="", password=password),
            Response(),
        )
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            name = conn.execute("SELECT name FROM accounts WHERE username = 'spacey'").fetchone()[0]
            conn.execute("UPDATE accounts SET status = '正常' WHERE username = 'spacey'")
        self.assertEqual(name, "名")
        self.assertEqual(self._login("spacey", password)[0]["status"], "success")
        with self.assertRaises(AuthenticationError):
            self._login("spacey", password.strip())

    def test_initial_admin_password_is_validated_only_when_the_account_is_created(self):
        # 已有账号库：残留的短引导密码不影响启动。
        with patch.dict("os.environ", {"INITIAL_ADMIN_PASSWORD": "short"}):
            app_storage.bootstrap_defaults()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute("DELETE FROM accounts")
        with patch.dict("os.environ", {"INITIAL_ADMIN_PASSWORD": "short"}), self.assertRaises(
            ValueError
        ):
            app_storage.bootstrap_defaults()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0], 0)
        app_storage.bootstrap_defaults()
        self.assertEqual(self._login_admin()["status"], "success")

    def test_account_cannot_change_its_own_role_status_or_delete_itself(self):
        principal = self._admin_principal()
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
        principal = self._admin_principal()
        assert principal.role_id is not None
        with self.assertRaises(AccessDeniedError) as error:
            role_router.update_role(
                principal.role_id,
                schemas.RolePermissionsRequest(
                    menu_permissions=list(principal.menu_permissions),
                    all_knowledge_bases=principal.all_knowledge_bases,
                    knowledge_base_ids=list(principal.knowledge_base_ids),
                ),
                principal,
            )
        self.assertEqual(error.exception.code, "cannot_edit_own_role")

    def test_limited_manager_cannot_grant_permissions_it_does_not_have(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, all_knowledge_bases, created_at) VALUES ('局部管理员', '', '[\"用户管理\", \"角色管理\"]', 0, ?)",
                (application_config.db_now(),),
            )
            limited_role_id = conn.execute(
                "SELECT id FROM roles WHERE name = '局部管理员'"
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, all_knowledge_bases, created_at) VALUES ('目标角色', '', '[]', 0, ?)",
                (application_config.db_now(),),
            )
            target_role_id = conn.execute(
                "SELECT id FROM roles WHERE name = '目标角色'"
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO accounts (username, name, password, status, role_id, created_at) VALUES ('manager', '局部管理员', ?, '正常', ?, ?)",
                (application_config.hash_password("password"), limited_role_id, application_config.db_now()),
            )
        manager = self._principal("manager", "password")
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
                "INSERT INTO roles (name, description, menu_permissions, all_knowledge_bases, created_at) VALUES ('局部管理员', '', '[\"用户管理\", \"角色管理\"]', 0, ?)",
                (application_config.db_now(),),
            )
            role_id = conn.execute("SELECT id FROM roles WHERE name = '局部管理员'").fetchone()[0]
            conn.execute(
                "INSERT INTO accounts (username, name, password, status, role_id, created_at) VALUES ('manager', '局部管理员', ?, '正常', ?, ?)",
                (application_config.hash_password("password"), role_id, application_config.db_now()),
            )
        return self._principal("manager", "password")

    def test_limited_manager_cannot_shrink_a_role_with_more_permissions(self):
        manager = self._limited_role_manager()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, all_knowledge_bases, created_at) VALUES ('平级角色', '', '[\"用户管理\", \"模型配置\"]', 0, ?)",
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
                "INSERT INTO roles (name, description, menu_permissions, all_knowledge_bases, created_at) VALUES ('下级角色', '', '[\"用户管理\"]', 0, ?)",
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
                "INSERT INTO roles (name, description, menu_permissions, all_knowledge_bases, created_at) VALUES ('用户管理员', '', '[\"用户管理\"]', 0, ?)",
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
        manager = self._principal("usermgr", "password")
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



class SessionLifecycleTest(AuthTestBase):
    def test_raw_token_is_never_stored_and_each_login_gets_a_new_one(self):
        _first, token_a = self._login()
        _second, token_b = self._login()

        self.assertNotEqual(token_a, token_b)
        self.assertEqual(len(token_a), 43)
        stored = {row[0] for row in self._session_rows()}
        self.assertEqual(
            stored, {hashlib.sha256(token.encode()).hexdigest() for token in (token_a, token_b)}
        )
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            dump = "\n".join(conn.iterdump())
        self.assertNotIn(token_a, dump)
        self.assertNotIn(token_b, dump)

    def test_stored_digest_cannot_be_replayed_as_a_cookie(self):
        self._login()
        (digest, _account), = self._session_rows()
        # 64 位十六进制摘要连格式都不符；截成 43 位也只会被再摘要一次而查不到。
        for forged in (digest, digest[:43]):
            with self.subTest(forged=forged), self.assertRaises(HTTPException) as error:
                auth_service.current_principal(fake_request(forged))
            self.assertEqual(error.exception.status_code, 401)

    def test_session_expiry_uses_the_server_clock_even_without_pruning(self):
        start = 1_900_000_000
        env = patch.dict("os.environ", {"AUTH_SESSION_TTL_SECONDS": "120"})
        env.start()
        self.addCleanup(env.stop)
        with patch.object(sessions, "now", return_value=start):
            _result, token = self._login()
        for moment, valid in ((start + 119, True), (start + 120, False), (start + 500, False)):
            with self.subTest(moment=moment), patch.object(sessions, "now", return_value=moment):
                if valid:
                    auth_service.current_principal(fake_request(token))
                else:
                    with self.assertRaises(HTTPException) as error:
                        auth_service.current_principal(fake_request(token))
                    self.assertEqual(error.exception.status_code, 401)
        # 过期行仍在库中，说明拒绝不依赖清理任务。
        self.assertEqual(len(self._session_rows()), 1)
        with patch.object(sessions, "now", return_value=start + 500):
            self.assertEqual(sessions.prune_expired(app_storage.DB_FILE), 1)

    def test_active_sessions_slide_up_to_the_absolute_cap(self):
        start = 1_900_000_000
        env = {"AUTH_SESSION_TTL_SECONDS": "3600", "AUTH_SESSION_MAX_AGE_SECONDS": "7200"}
        with patch.dict("os.environ", env):
            with patch.object(sessions, "now", return_value=start):
                _result, token = self._login()
            # 不足续期间隔时不写库。
            with patch.object(sessions, "now", return_value=start + 60):
                auth_service.current_principal(fake_request(token))
            self.assertEqual(self._expiry(), (start + 3600, start + 7200))
            # 空闲时限内有请求就向后滑动。
            with patch.object(sessions, "now", return_value=start + 3000):
                auth_service.current_principal(fake_request(token))
            self.assertEqual(self._expiry()[0], start + 6600)
            # 再滑动也不超过绝对上限，到点即失效。
            with patch.object(sessions, "now", return_value=start + 6000):
                auth_service.current_principal(fake_request(token))
            self.assertEqual(self._expiry()[0], start + 7200)
            with patch.object(sessions, "now", return_value=start + 7200), self.assertRaises(
                HTTPException
            ):
                auth_service.current_principal(fake_request(token))

    def test_renewal_failure_does_not_fail_the_request(self):
        start = 1_900_000_000
        with patch.object(sessions, "now", return_value=start):
            _result, token = self._login()
        real_connect = sessions.application_db.connect

        def connect(db_file, **kwargs):
            # 只让续期用的短超时连接失败，读取主体的连接照常。
            if kwargs.get("timeout") == 1.0:
                raise sqlite3.OperationalError("database is locked")
            return real_connect(db_file, **kwargs)

        with patch.object(sessions, "now", return_value=start + 3000), patch.object(
            sessions.application_db, "connect", side_effect=connect
        ), self.assertLogs(sessions.logger, "WARNING"):
            self.assertEqual(auth_service.current_principal(fake_request(token)).username, ADMIN)
        self.assertEqual(self._expiry()[0], start + 28800)

    def test_each_account_keeps_at_most_the_configured_number_of_sessions(self):
        tokens = []
        with patch.dict("os.environ", {"AUTH_MAX_SESSIONS_PER_ACCOUNT": "2"}):
            for moment in (1_900_000_000, 1_900_000_001, 1_900_000_002):
                with patch.object(sessions, "now", return_value=moment):
                    tokens.append(self._login()[1])
        self.assertEqual(len(self._session_rows()), 2)
        with self.assertRaises(HTTPException):
            auth_service.current_principal(fake_request(tokens[0]))
        for token in tokens[1:]:
            with patch.object(sessions, "now", return_value=1_900_000_003):
                auth_service.current_principal(fake_request(token))

    def _expiry(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            return conn.execute(
                "SELECT expires_at, absolute_expires_at FROM auth_sessions"
            ).fetchone()

    def test_session_ttl_setting_has_a_60_second_floor(self):
        with patch.object(sessions, "now", return_value=1_000), patch.dict(
            "os.environ", {"AUTH_SESSION_TTL_SECONDS": "5"}
        ):
            self._login()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            created, expires = conn.execute(
                "SELECT created_at, expires_at FROM auth_sessions"
            ).fetchone()
        self.assertEqual(expires - created, 60)

    def test_sessions_survive_a_process_restart(self):
        _result, token = self._login()
        from services.core import container as application_container

        application_container._default_container = None
        app_storage.initialize_database()
        self.assertEqual(auth_service.current_principal(fake_request(token)).username, ADMIN)

    def test_logout_deletes_the_session_and_a_copied_cookie_stops_working(self):
        async def exercise():
            async with api_client() as browser, api_client() as other_browser:
                login = await self._http_login(browser)
                token = cookie_token(login)
                await self._http_login(other_browser)
                logout = await browser.post("/api/logout")
                again = await browser.post("/api/logout")
                async with api_client(cookies={sessions.COOKIE_NAME: token}) as copied:
                    replay = await copied.get("/api/me")
                return logout, again, replay, await other_browser.get("/api/me")

        logout, again, replay, other = run(exercise)
        self.assertEqual(logout.status_code, 204)
        self.assertEqual(logout.headers["cache-control"], "no-store")
        attributes = set_cookie_attributes(logout.headers["set-cookie"])
        self.assertEqual(attributes["max-age"], "0")
        self.assertEqual(attributes["path"], "/api")
        self.assertIn("httponly", attributes)
        self.assertEqual(attributes["samesite"].lower(), "lax")
        self.assertEqual(again.status_code, 204)
        self.assertEqual(replay.status_code, 401)
        # 另一个浏览器的独立会话不受影响。
        self.assertEqual(other.status_code, 200)
        self.assertEqual(len(self._session_rows()), 1)

    def test_logout_without_or_with_malformed_or_expired_cookie_is_204(self):
        start = 1_900_000_000
        with patch.object(sessions, "now", return_value=start):
            _result, token = self._login()

        async def exercise():
            responses = []
            for cookies in ({}, {sessions.COOKIE_NAME: "bad!"}, {sessions.COOKIE_NAME: token}):
                async with api_client(cookies=cookies) as client:
                    responses.append(await client.post("/api/logout"))
            return responses

        with patch.object(sessions, "now", return_value=start + 10**6):
            responses = run(exercise)
        self.assertEqual([response.status_code for response in responses], [204, 204, 204])
        self.assertEqual(self._session_rows(), [])

    def test_login_replaces_the_session_presented_by_the_same_browser(self):
        user_id = self._add_user("second")
        _result, admin_token = self._login()
        _result, new_token = self._login("second", "user-password", token=admin_token)

        self.assertEqual([row[1] for row in self._session_rows()], [user_id])
        with self.assertRaises(HTTPException):
            auth_service.current_principal(fake_request(admin_token))
        self.assertEqual(
            auth_service.current_principal(fake_request(new_token)).username, "second"
        )

    def test_failed_login_keeps_the_existing_session(self):
        _result, token = self._login()
        with self.assertRaises(AuthenticationError):
            self._login(ADMIN, "wrong-password", token=token)
        self.assertEqual(auth_service.current_principal(fake_request(token)).username, ADMIN)

    def test_malformed_or_oversized_cookies_are_401_without_clearing_the_cookie(self):
        cookies = ("", "short", "x" * 43 + "!", "a" * 44, "a" * 5000, "a" * 43, "é" * 43)

        async def exercise():
            responses = []
            for value in cookies:
                headers = {"cookie": f"{sessions.COOKIE_NAME}={value}".encode()}
                async with api_client(headers=headers) as client:
                    responses.append(await client.get("/api/me"))
            async with api_client() as client:
                responses.append(await client.get("/api/me"))
            return responses

        for response in run(exercise):
            self.assertEqual(response.status_code, 401, response.text)
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertNotIn("set-cookie", response.headers)

    def test_bearer_header_is_not_accepted(self):
        _result, token = self._login()

        async def exercise():
            async with api_client(headers={"Authorization": f"Bearer {token}"}) as client:
                return await client.get("/api/me")

        self.assertEqual(run(exercise).status_code, 401)

    def test_database_errors_are_not_reported_as_unauthenticated(self):
        _result, token = self._login()
        with patch.object(
            auth_service.application_db,
            "connect",
            side_effect=sqlite3.OperationalError("database is locked"),
        ), self.assertRaises(sqlite3.OperationalError):
            auth_service.current_principal(fake_request(token))

    def test_session_cookie_attributes(self):
        async def exercise():
            async with api_client() as client:
                return await self._http_login(client)

        with patch.dict("os.environ", {"AUTH_SESSION_TTL_SECONDS": "3600"}):
            default = run(exercise)
        with patch.dict("os.environ", {"AUTH_COOKIE_SECURE": "true"}):
            secure = run(exercise)

        self.assertNotIn("access_token", default.json())
        header = default.headers["set-cookie"]
        self.assertTrue(header.startswith(f"{sessions.COOKIE_NAME}="))
        attributes = set_cookie_attributes(header)
        self.assertIn("httponly", attributes)
        self.assertEqual(attributes["samesite"].lower(), "lax")
        self.assertEqual(attributes["path"], "/api")
        # Cookie 活到绝对上限（默认 7 天），空闲过期由服务端判断。
        self.assertEqual(attributes["max-age"], "604800")
        self.assertNotIn("domain", attributes)
        self.assertNotIn("secure", attributes)
        self.assertIn("secure", set_cookie_attributes(secure.headers["set-cookie"]))


class CredentialRevocationTest(AuthTestBase):
    def _change_password(self, token, old, new):
        principal = auth_service.current_principal(fake_request(token))
        response = Response()
        result = auth.change_password(
            schemas.PasswordChangeRequest(old_password=old, new_password=new),
            fake_request(token),
            response,
            principal,
        )
        return result, response

    def test_password_change_revokes_every_session_and_clears_the_cookie(self):
        async def exercise():
            async with api_client() as browser, api_client() as phone:
                await self._http_login(browser)
                await self._http_login(phone)
                changed = await browser.post(
                    "/api/change_password",
                    json={
                        "old_password": application_config.initial_admin_password(),
                        "new_password": "replacement-password",
                    },
                )
                return changed, await browser.get("/api/me"), await phone.get("/api/me")

        changed, browser_me, phone_me = run(exercise)
        self.assertEqual(changed.status_code, 200, changed.text)
        self.assertEqual(changed.json()["status"], "success")
        self.assertEqual(changed.headers["cache-control"], "no-store")
        self.assertEqual(set_cookie_attributes(changed.headers["set-cookie"])["max-age"], "0")
        self.assertEqual((browser_me.status_code, phone_me.status_code), (401, 401))
        self.assertEqual(self._session_rows(), [])
        self.assertEqual(self._login(ADMIN, "replacement-password")[0]["status"], "success")
        with self.assertRaises(AuthenticationError):
            self._login_admin()

    def test_wrong_old_password_changes_nothing_and_keeps_the_cookie(self):
        _result, token = self._login()
        before = self._stored_password()

        async def exercise():
            async with api_client(cookies={sessions.COOKIE_NAME: token}) as client:
                return await client.post(
                    "/api/change_password",
                    json={"old_password": "wrong-password", "new_password": "replacement-password"},
                )

        response = run(exercise)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_current_password")
        self.assertNotIn("set-cookie", response.headers)
        self.assertEqual(self._stored_password(), before)
        self.assertEqual(len(self._session_rows()), 1)

    def test_failure_while_revoking_sessions_rolls_back_the_password(self):
        _result, token = self._login()
        before = self._stored_password()
        with patch.object(
            sessions, "delete_for_account", side_effect=sqlite3.OperationalError("disk full")
        ), self.assertRaises(sqlite3.OperationalError):
            self._change_password(
                token, application_config.initial_admin_password(), "replacement-password"
            )
        self.assertEqual(self._stored_password(), before)
        self.assertEqual(auth_service.current_principal(fake_request(token)).username, ADMIN)

    def test_conditional_update_refuses_a_password_changed_meanwhile(self):
        _result, token = self._login()
        digest = sessions.token_hash(token)
        assert digest is not None
        old_hash = self._stored_password()
        concurrent_hash = application_config.hash_password("concurrent-password")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute("UPDATE accounts SET password = ? WHERE username = ?", (concurrent_hash, ADMIN))

        self.assertFalse(
            accounts.replace_password(
                app_storage.DB_FILE,
                account_id=self._admin_id(),
                session_digest=digest,
                expected_hash=old_hash,
                new_hash=application_config.hash_password("late-password"),
            )
        )
        self.assertEqual(self._stored_password(), concurrent_hash)
        self.assertEqual(len(self._session_rows()), 1)

    def test_concurrent_password_changes_do_not_overwrite_each_other(self):
        tokens = [self._login()[1] for _ in range(2)]
        real_hash = application_config.hash_password
        # 两个请求都在事务外校验过原密码并算好新哈希后，才同时进入写事务。
        both_hashed = threading.Barrier(2, timeout=30)

        def hash_then_wait(password):
            hashed = real_hash(password)
            both_hashed.wait()
            return hashed

        outcomes: dict[str, object] = {}

        def change(token, new_password):
            try:
                self._change_password(
                    token, application_config.initial_admin_password(), new_password
                )
                outcomes[new_password] = "ok"
            except Exception as exc:
                outcomes[new_password] = exc

        with patch.object(application_config, "hash_password", side_effect=hash_then_wait):
            threads = [
                threading.Thread(target=change, args=(token, f"new-password-{index}"))
                for index, token in enumerate(tokens)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        winners = [password for password, outcome in outcomes.items() if outcome == "ok"]
        losers = [outcome for outcome in outcomes.values() if outcome != "ok"]
        self.assertEqual(len(winners), 1, outcomes)
        self.assertEqual(len(losers), 1)
        self.assertIsInstance(losers[0], ConflictError)
        self.assertEqual(losers[0].code, "credentials_changed")  # type: ignore[attr-defined]
        self.assertTrue(application_config.verify_password(winners[0], self._stored_password()))
        self.assertEqual(self._session_rows(), [])

    def test_login_that_races_a_password_change_does_not_create_a_session(self):
        real_create = sessions.create
        _result, token = self._login()

        def change_then_create(*args, **kwargs):
            # bcrypt 已用旧密码通过，写事务开始前密码被改。
            self._change_password(
                token, application_config.initial_admin_password(), "replacement-password"
            )
            return real_create(*args, **kwargs)

        with patch.object(sessions, "create", side_effect=change_then_create), self.assertRaises(
            AuthenticationError
        ) as error:
            self._login_admin()
        self.assertEqual(error.exception.code, "invalid_credentials")
        self.assertEqual(self._session_rows(), [])

    def test_login_that_races_a_disable_does_not_create_a_session(self):
        user_id = self._add_user("member")
        real_create = sessions.create

        def disable_then_create(*args, **kwargs):
            accounts.update_user_status(app_storage.DB_FILE, user_id, "停用")
            return real_create(*args, **kwargs)

        with patch.object(sessions, "create", side_effect=disable_then_create), self.assertRaises(
            AuthenticationError
        ):
            self._login("member", "user-password")
        self.assertEqual(self._session_rows(), [])

    def test_disabling_revokes_sessions_and_reenabling_does_not_revive_them(self):
        user_id = self._add_user("member")
        _result, token = self._login("member", "user-password")
        admin = self._admin_principal()

        user_router.update_user_status(user_id, schemas.UserStatusRequest(status="停用"), admin)
        self.assertEqual(self._session_rows(user_id), [])
        user_router.update_user_status(user_id, schemas.UserStatusRequest(status="正常"), admin)

        with self.assertRaises(HTTPException) as error:
            auth_service.current_principal(fake_request(token))
        self.assertEqual(error.exception.status_code, 401)
        # 管理员自己的会话不受影响。
        self.assertEqual(len(self._session_rows(self._admin_id())), 1)

    def test_moving_back_to_review_also_revokes_sessions(self):
        user_id = self._add_user("member")
        self._login("member", "user-password")
        accounts.update_user_status(app_storage.DB_FILE, user_id, "审核中")
        self.assertEqual(self._session_rows(user_id), [])

    def test_deleting_an_account_cascades_to_its_sessions(self):
        user_id = self._add_user("member")
        _result, token = self._login("member", "user-password")
        user_router.delete_user(user_id, self._admin_principal())

        self.assertEqual(self._session_rows(user_id), [])
        # 同名新账号不能继承旧会话。
        self._add_user("member")
        with self.assertRaises(HTTPException):
            auth_service.current_principal(fake_request(token))


class PasswordChangeRateLimitTest(AuthTestBase):
    def setUp(self):
        super().setUp()
        self.now = 1000.0
        self.limiter = rate_limit.AuthRateLimiter(
            login_failures=3,
            login_window_seconds=600,
            client_failures=100,
            client_window_seconds=600,
            register_attempts=10,
            register_window_seconds=600,
            clock=lambda: self.now,
        )
        patcher = patch.object(rate_limit, "_limiter", self.limiter)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.token = self._login()[1]

    def _change(self, old, new="replacement-password"):
        return auth.change_password(
            schemas.PasswordChangeRequest(old_password=old, new_password=new),
            fake_request(self.token),
            Response(),
            auth_service.current_principal(fake_request(self.token)),
        )

    def test_five_wrong_old_passwords_block_the_account_for_15_minutes(self):
        for _ in range(5):
            with self.assertRaises(InvalidStateError):
                self._change("wrong-password")
        with patch.object(application_config, "verify_password") as verify, self.assertRaises(
            RateLimitedError
        ) as locked:
            self._change(application_config.initial_admin_password())
        verify.assert_not_called()
        self.assertEqual(locked.exception.code, "password_change_rate_limited")
        self.assertEqual(locked.exception.retry_after_seconds, 900)
        # 与登录窗口互相独立：改密被锁不影响登录。
        self.assertEqual(self._login_admin()["status"], "success")

        self.now += 901
        self.assertEqual(self._change(application_config.initial_admin_password())["status"], "success")

    def test_success_clears_the_failure_count_and_login_failures_do_not_count(self):
        for _ in range(3):
            with self.assertRaises(AuthenticationError):
                self._login(ADMIN, "wrong-password")
        for _ in range(4):
            with self.assertRaises(InvalidStateError):
                self._change("wrong-password")
        self.now += 601  # 登录窗口过期，改密窗口仍在
        self._change(application_config.initial_admin_password(), "second-password-1")
        self.token = self._login(ADMIN, "second-password-1")[1]
        for _ in range(4):
            with self.assertRaises(InvalidStateError):
                self._change("wrong-password")
        self.assertEqual(self._change("second-password-1", "third-password-1")["status"], "success")

    def test_http_response_is_429_with_retry_after(self):
        async def exercise():
            async with api_client(cookies={sessions.COOKIE_NAME: self.token}) as client:
                for _ in range(6):
                    response = await client.post(
                        "/api/change_password",
                        json={"old_password": "wrong-pass", "new_password": "replacement-password"},
                    )
                return response

        response = run(exercise)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "900")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn("set-cookie", response.headers)


class WriteHeaderTest(AuthTestBase):
    def _write_routes(self):
        """从 OpenAPI 文档枚举全部 /api 写路由，新增路由自动纳入检查。"""
        for path, operations in api.app.openapi()["paths"].items():
            if not path.startswith("/api"):
                continue
            for method in sorted(operations):
                if method.upper() not in {"GET", "HEAD", "OPTIONS"}:
                    yield method.upper(), path

    def test_every_write_route_rejects_requests_without_the_header(self):
        routes = list(self._write_routes())
        paths = {path for _method, path in routes}
        for required in ("/api/login", "/api/register", "/api/logout", "/api/change_password"):
            self.assertIn(required, paths)
        _result, token = self._login()
        before = self._stored_password()

        async def exercise():
            responses = []
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api.app),
                base_url="http://test",
                cookies={sessions.COOKIE_NAME: token},
            ) as client:
                for method, path in routes:
                    concrete = path
                    for name in ("kb_id", "document_id", "session_id", "provider", "job_id"):
                        concrete = concrete.replace("{" + name + "}", "x")
                    concrete = concrete.replace("{user_id}", "1").replace("{role_id}", "1")
                    concrete = concrete.replace("{message_id}", "1")
                    response = await client.request(
                        method,
                        concrete,
                        json={
                            "username": ADMIN,
                            "password": application_config.initial_admin_password(),
                        },
                        headers={"X-RAGLoom-Request": "0"} if path == "/api/logout" else None,
                    )
                    responses.append((method, path, response))
            return responses

        for method, path, response in run(exercise):
            with self.subTest(method=method, path=path):
                self.assertEqual(response.status_code, 403, response.text)
                self.assertEqual(
                    response.json(), {"detail": response.json()["detail"], "code": "csrf_rejected"}
                )
        # 被拒绝的写请求没有任何副作用：会话、密码都还在。
        self.assertEqual(len(self._session_rows()), 1)
        self.assertEqual(self._stored_password(), before)

    def test_safe_methods_and_correct_header_pass(self):
        async def exercise():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api.app), base_url="http://test"
            ) as client:
                health = await client.get("/api/health")
                options = await client.options("/api/login")
                login = await client.post(
                    "/api/login",
                    json={"username": ADMIN, "password": application_config.initial_admin_password()},
                    headers=WRITE_HEADERS,
                )
                return health, options, login, await client.get("/api/me")

        health, options, login, me = run(exercise)
        self.assertEqual(health.status_code, 200)
        self.assertNotEqual(options.status_code, 403)
        self.assertNotIn("access-control-allow-origin", options.headers)
        self.assertEqual(login.status_code, 200)
        self.assertEqual(me.status_code, 200)


class ValidationErrorOutputTest(AuthTestBase):
    SECRET = "SECRET-MARKER-9f2c"

    def _post(self, path, payload):
        async def exercise():
            async with api_client() as client:
                await self._http_login(client)
                return await client.post(path, json=payload)

        return run(exercise)

    def test_auth_and_model_validation_errors_never_echo_the_input(self):
        cases = [
            ("/api/login", {"username": "admin", "password": self.SECRET * 40}),
            ("/api/login", {"username": "admin", "password": self.SECRET, "extra": self.SECRET}),
            ("/api/register", {"username": "u", "name": "n", "phone": self.SECRET * 3, "password": self.SECRET}),
            ("/api/change_password", {"old_password": self.SECRET, "new_password": self.SECRET * 40}),
            (
                "/api/models",
                {"provider": "p", "api_key": self.SECRET * 400, "base_url": "https://x.test", "default_model": "m"},
            ),
        ]
        for path, payload in cases:
            with self.subTest(path=path):
                response = self._post(path, payload)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNotIn(self.SECRET, response.text)
                for item in response.json()["detail"]:
                    self.assertLessEqual(set(item), {"loc", "type", "msg", "ctx"})
                    self.assertLessEqual(set(item.get("ctx", {})), {"min_length", "max_length"})

        too_long = self._post("/api/login", {"username": "admin", "password": "p" * 300})
        self.assertEqual(too_long.json()["detail"][0]["ctx"], {"max_length": 256})
        self.assertEqual(too_long.headers["cache-control"], "no-store")

    def test_other_routes_keep_the_default_validation_output(self):
        response = self._post("/api/chat", {"question": self.SECRET})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("input", response.json()["detail"][0])

    def test_failed_login_responses_are_not_cacheable(self):
        async def exercise():
            async with api_client() as client:
                return await client.post(
                    "/api/login", json={"username": ADMIN, "password": "wrong-password"}
                )

        response = run(exercise)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn("set-cookie", response.headers)


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

    def test_reserve_checks_and_counts_atomically_and_release_returns_the_slot(self):
        window = self._window()
        slots = [window.reserve("alice") for _ in range(4)]

        self.assertEqual([wait for _slot, wait in slots], [0, 0, 0, 60])
        self.assertIsNone(slots[3][0])
        window.release("alice", slots[0][0])
        self.assertEqual(window.retry_after("alice"), 0)
        window.release("alice", None)
        window.release("missing", 1.0)

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
            schemas.LoginRequest(username=username, password=password),
            fake_request(),
            Response(),
            client,
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

    def test_concurrent_wrong_passwords_cannot_exceed_the_limit(self):
        started = threading.Barrier(8)
        real_verify = application_config.verify_password

        def slow_verify(*args, **kwargs):
            # 慢速的 bcrypt：所有请求都会在任何一个记录失败之前通过检查（修复前）。
            time.sleep(0.05)
            return real_verify(*args, **kwargs)

        def attempt():
            started.wait()
            try:
                self._login("wrong")
            except (AuthenticationError, RateLimitedError) as exc:
                return type(exc)

        with patch.object(application_config, "verify_password", side_effect=slow_verify):
            with concurrent.futures.ThreadPoolExecutor(8) as pool:
                results = list(pool.map(lambda _: attempt(), range(8)))

        self.assertEqual(results.count(AuthenticationError), 3)
        self.assertEqual(results.count(RateLimitedError), 5)

    def test_correct_password_returns_the_reserved_client_slot(self):
        for _ in range(10):
            self._login(self._admin_password())

        self.assertEqual(self.limiter._by_client.retry_after("10.0.0.1"), 0)

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
            Response(),
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
                Response(),
                "10.0.0.7",
            )

        with self.assertRaises(RateLimitedError) as locked:
            auth.register_user(
                schemas.RegistrationRequest(
                    username="new-3", name="新用户", phone="", password="new-pass-123"
                ),
                Response(),
                "10.0.0.7",
            )
        self.assertEqual(locked.exception.code, "registration_rate_limited")
        auth.register_user(
            schemas.RegistrationRequest(
                username="other", name="其他", phone="", password="new-pass-123"
            ),
            Response(),
            "10.0.0.8",
        )

    def test_empty_client_address_skips_address_based_limits(self):
        for index in range(10):
            with self.assertRaises(AuthenticationError):
                self._login("wrong", username=f"user-{index}", client="")

    def test_http_response_is_429_with_retry_after_header(self):
        async def exercise():
            async with api_client() as client:
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
        self.kb_id = add_knowledge_bases("技术部")[0]

    def _chat(self, count):
        async def events():
            yield "data: {}\n\n"

        async def fake_chat(*_args, **_kwargs):
            return SimpleNamespace(
                events=events(),
                refresh_summary=lambda: asyncio.sleep(0),
                abandon=lambda: asyncio.sleep(0),
            )

        async def run():
            async with api_client() as client:
                await client.post(
                    "/api/login",
                    json={
                        "username": application_config.INITIAL_ADMIN_USERNAME,
                        "password": application_config.initial_admin_password(),
                    },
                )
                return [
                    await client.post(
                        "/api/chat", json={"question": "问题", "kb_id": self.kb_id}
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


    def test_disconnect_before_the_first_event_frees_the_slot(self):
        """客户端在首包前断开时生成器可能一次都不被迭代，名额和收尾仍要立即完成。"""
        calls = []

        async def events():
            calls.append("iterated")
            yield "data: {}\n\n"

        async def abandon():
            calls.append("abandon")

        async def refresh_summary():
            calls.append("summary")
            # 摘要刷新时名额必须已经释放，不能让摘要调用占用户的并发名额。
            rate_limit.get_chat_request_limiter().acquire(1)()

        stream = SimpleNamespace(
            events=events(), refresh_summary=refresh_summary, abandon=abandon
        )
        with patch.dict("os.environ", {"CHAT_USER_MAX_CONCURRENT": "1", "CHAT_USER_PER_MINUTE": "10"}):
            rate_limit.reset_auth_rate_limiter()
            release = rate_limit.get_chat_request_limiter().acquire(1)
            response = chat_router._ChatStreamResponse(stream, release)
            sent = []

            async def receive():
                return {"type": "http.disconnect"}

            async def send(message):
                sent.append(message["type"])
                await asyncio.sleep(0)

            asyncio.run(
                response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
            )
            self.assertEqual(calls, ["abandon", "summary"])
            rate_limit.get_chat_request_limiter().acquire(1)()


if __name__ == "__main__":
    unittest.main()
