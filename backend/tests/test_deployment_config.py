import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import deployment_smoke
from services.core.settings import ApplicationSettings

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
CONFIGURATION_DOC = PROJECT_DIR / "docs/development/配置项.md"
DEPLOYMENT_SECTION = "## 部署与构建"
LONG_RUNNING_SERVICES = ("backend", "milvus", "frontend")
# 公开仓库不含 docs/，依赖配置文档的测试只在开发仓库里运行。
requires_configuration_doc = unittest.skipUnless(CONFIGURATION_DOC.exists(), "配置文档不在当前仓库中")


def _service_block(compose: str, name: str) -> str:
    """返回 docker-compose.yml 中某个服务的配置块，到下一个服务为止。"""
    marker = f"\n  {name}:\n"
    start = compose.index(marker) + len(marker)
    following = re.search(r"\n  [\w-]+:\n", compose[start:])
    return compose[start:] if following is None else compose[start:][: following.start()]


def _documented_variables() -> tuple[set[str], set[str]]:
    """返回配置文档表格首列中的全部变量，以及“部署与构建”一节中只给 Compose 用的变量。"""
    documented: set[str] = set()
    deployment_only: set[str] = set()
    section = ""
    for line in CONFIGURATION_DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line
        elif line.startswith("| `"):
            names = set(re.findall(r"`([A-Z][A-Z0-9_]+)`", line.split("|")[1]))
            documented |= names
            if section == DEPLOYMENT_SECTION:
                deployment_only |= names
    return documented, deployment_only


class DeploymentConfigTest(unittest.TestCase):
    def test_runtime_image_excludes_sources_it_does_not_need(self):
        dockerfile = (BACKEND_DIR / "Dockerfile").read_text(encoding="utf-8")
        dockerignore = (BACKEND_DIR / ".dockerignore").read_text(encoding="utf-8")

        self.assertNotIn("COPY . .", dockerfile)
        for excluded_path in ("tests/", "evaluation/", "scripts/", "storage/"):
            self.assertIn(excluded_path, dockerignore)

    def test_backend_runs_as_an_unprivileged_user_with_a_persistent_model_cache(self):
        dockerfile = (BACKEND_DIR / "Dockerfile").read_text(encoding="utf-8")
        compose = (PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("USER app", dockerfile)
        self.assertIn('user: "${APP_UID:-1000}:${APP_GID:-1000}"', compose)
        self.assertIn("paddle_models:/home/app/.paddlex/official_models", compose)
        self.assertNotIn("/root/.paddlex", compose)

    def test_backend_is_only_reachable_from_the_host_loopback(self):
        compose = (PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn('- "127.0.0.1:8000:8000"', _service_block(compose, "backend"))

    def test_long_running_services_rotate_their_logs(self):
        compose = (PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8")

        for name in LONG_RUNNING_SERVICES:
            self.assertIn("logging: *default-logging", _service_block(compose, name))
        self.assertIn('max-size: "10m"', compose)

    def test_backend_reads_dot_env_and_pins_container_layout(self):
        compose = (PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8")
        backend = _service_block(compose, "backend")

        self.assertRegex(backend, r"env_file:\n\s+- path: \.env\n\s+required: false")
        # environment 优先于 env_file，路径和 Milvus 地址不能被 .env 改掉。
        for line in (
            "STORAGE_DIR: /app/storage",
            "MINERU_RESULTS_DIR: /app/storage/mineru_results",
            "MILVUS_URI: http://milvus:19530",
            "AUTH_TRUST_X_REAL_IP: ${AUTH_TRUST_X_REAL_IP:-true}",
        ):
            self.assertIn(f"      {line}\n", backend)

    @requires_configuration_doc
    def test_configuration_doc_lists_exactly_the_settings_the_backend_reads(self):
        settings_source = (BACKEND_DIR / "services/core/settings.py").read_text(encoding="utf-8")
        read_by_backend = set(
            re.findall(r'(?:os\.getenv|_integer|_flag)\(\s*"([A-Z][A-Z0-9_]+)"', settings_source)
        )
        documented, deployment_only = _documented_variables()

        self.assertEqual(documented - deployment_only, read_by_backend)
        self.assertFalse(deployment_only & read_by_backend)

    @requires_configuration_doc
    def test_deployment_variables_are_used_by_compose_and_compose_uses_no_undocumented_ones(self):
        compose = (PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8")
        referenced = set(re.findall(r"\$\{([A-Z][A-Z0-9_]+)", compose))
        documented, deployment_only = _documented_variables()

        self.assertLessEqual(deployment_only, referenced)
        self.assertLessEqual(referenced, documented)

    @requires_configuration_doc
    def test_env_example_only_contains_documented_variables(self):
        env_example = (PROJECT_DIR / ".env.example").read_text(encoding="utf-8")
        names = re.findall(r"^([A-Z][A-Z0-9_]*)=", env_example, flags=re.MULTILINE)
        documented, _ = _documented_variables()

        self.assertIn("MODEL_API_KEY_ENCRYPTION_KEY", names)
        self.assertEqual(len(names), len(set(names)))
        self.assertLessEqual(set(names), documented)

    def test_frontend_waits_for_a_healthy_backend(self):
        compose = (PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("http://localhost:8000/api/health", compose)
        self.assertRegex(compose, r"backend:\n\s+condition: service_healthy")

    def test_deployment_hardening_settings_are_in_place(self):
        compose = (PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8")
        dockerfile = (BACKEND_DIR / "Dockerfile").read_text(encoding="utf-8")
        smoke = (PROJECT_DIR / "scripts" / "smoke-test.sh").read_text(encoding="utf-8")

        # 默认口令、关闭沙箱和可变的 Milvus 镜像都是发布前必须挡住的回退。
        self.assertNotIn("admin888", compose)
        self.assertNotIn("seccomp:unconfined", compose)
        self.assertRegex(compose, r"milvusdb/milvus:v[\d.]+@sha256:[0-9a-f]{64}")
        # 每条 pip install 都必须校验 hash，两个锁文件都要带上。
        self.assertIn("--require-hashes", dockerfile)
        self.assertEqual(
            dockerfile.count("pip install "), dockerfile.count("pip install --require-hashes ")
        )
        for lock_name in ("requirements.lock", "requirements-check.lock"):
            self.assertIn("--hash=sha256:", (BACKEND_DIR / lock_name).read_text(encoding="utf-8"))
        # 冒烟脚本只经宿主机回环访问，不依赖对外暴露的端口。
        self.assertIn("http://127.0.0.1:8000/api/health", smoke)
        self.assertIn("http://127.0.0.1:8080/", smoke)
        # 依赖只装在镜像里，冒烟脚本不能改用宿主机的解释器。
        self.assertIn("docker compose run --rm --no-deps", smoke)
        self.assertNotIn("python3 scripts/deployment_smoke.py", smoke)

    def test_direct_dependencies_are_pinned_and_installed_from_the_lock_file(self):
        dockerfile = (BACKEND_DIR / "Dockerfile").read_text(encoding="utf-8")
        pyproject = (BACKEND_DIR / "pyproject.toml").read_text(encoding="utf-8")
        direct_dependencies = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]

        self.assertIn("COPY requirements.lock .", dockerfile)
        self.assertFalse((BACKEND_DIR / "uv.lock").exists())
        pinned = re.findall(r'^\s+"([^"]+)",$', direct_dependencies, flags=re.MULTILINE)
        self.assertTrue(pinned)
        self.assertTrue(all("==" in pin for pin in pinned), pinned)

    def test_milvus_runs_as_a_single_healthy_standalone_container(self):
        compose = (PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8")
        milvus = _service_block(compose, "milvus")

        self.assertIn("DEPLOY_MODE: STANDALONE", milvus)
        self.assertIn(":9091/healthz", milvus)
        self.assertIn("condition: service_healthy", _service_block(compose, "backend"))

    def test_optional_retrieval_steps_are_off_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings.from_env()

        self.assertFalse(settings.chat_retrieval_planning)
        self.assertFalse(settings.chat_query_variant)
        self.assertFalse(settings.chat_agentic_retrieval)
        self.assertFalse(settings.auth_trust_x_real_ip)

    def test_chat_proxy_streams_without_buffering(self):
        nginx_config = (PROJECT_DIR / "frontend" / "nginx.conf").read_text(encoding="utf-8")
        chat_location = nginx_config.split("location = /api/chat {", 1)[1].split("}", 1)[0]

        self.assertIn("proxy_pass http://backend:8000/api/chat;", chat_location)
        self.assertIn("proxy_http_version 1.1;", chat_location)
        self.assertIn('proxy_set_header Connection "";', chat_location)
        self.assertIn("proxy_buffering off;", chat_location)
        self.assertIn("proxy_cache off;", chat_location)
        self.assertIn("proxy_read_timeout 300s;", chat_location)

    def test_deployment_smoke_submits_the_plaintext_password_contract(self):
        with patch.object(
            deployment_smoke,
            "request_json",
            return_value={"access_token": "test-token"},
        ) as request:
            token = deployment_smoke.login(
                "http://127.0.0.1:8000", "admin", "test-password"
            )

        self.assertEqual(token, "test-token")
        request.assert_called_once_with(
            "http://127.0.0.1:8000",
            "/api/login",
            method="POST",
            payload={"username": "admin", "password": "test-password"},
        )

    def test_local_check_script_runs_backend_and_frontend_regressions(self):
        check_script = (PROJECT_DIR / "scripts" / "check.sh").read_text(encoding="utf-8")

        for step in ("backend-check", "frontend-check", "unittest discover -s tests"):
            self.assertIn(step, check_script)
        # 检查工具来自检查镜像，不能在宿主机上临时拉取。
        self.assertNotIn("uvx", check_script)

    def test_static_checks_cover_the_same_backend_packages(self):
        pyproject = (BACKEND_DIR / "pyproject.toml").read_text(encoding="utf-8")
        compose = (PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8")
        ci = (PROJECT_DIR / ".github/workflows/ci.yml").read_text(encoding="utf-8")

        pyright_include = pyproject.split("[tool.pyright]", 1)[1].split("include = [", 1)[1]
        targets = re.findall(r'"([^"]+)"', pyright_include.split("]", 1)[0])
        self.assertTrue(targets)
        # ruff 与 pyright 覆盖同一批源码；按目录列出，新增模块自动进入检查范围。
        for source in (compose, ci):
            ruff_lines = [line for line in source.splitlines() if "ruff check" in line]
            self.assertEqual(len(ruff_lines), 1)
            for target in targets:
                self.assertIn(target, ruff_lines[0])
