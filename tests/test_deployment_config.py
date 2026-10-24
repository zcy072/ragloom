import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import deployment_smoke


class DeploymentConfigTest(unittest.TestCase):
    def test_backend_image_copies_only_runtime_sources(self):
        project_dir = Path(__file__).resolve().parents[1]
        dockerfile = (project_dir / "Dockerfile").read_text(encoding="utf-8")
        dockerignore = (project_dir / ".dockerignore").read_text(encoding="utf-8")

        self.assertNotIn("COPY . .", dockerfile)
        self.assertIn("COPY api.py app_storage.py ./", dockerfile)
        self.assertIn("COPY routers ./routers", dockerfile)
        self.assertIn("COPY services ./services", dockerfile)
        for excluded_path in (
            ".uv-tools/",
            "frontend/",
            "tests/",
            "docs/",
        ):
            self.assertIn(excluded_path, dockerignore)

    def test_backend_runs_as_the_configured_host_user(self):
        project_dir = Path(__file__).resolve().parents[1]
        dockerfile = (project_dir / "Dockerfile").read_text(encoding="utf-8")
        compose = (project_dir / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("ARG APP_UID=1000", dockerfile)
        self.assertIn("ARG APP_GID=1000", dockerfile)
        self.assertIn("USER app", dockerfile)
        self.assertIn('user: "${APP_UID:-1000}:${APP_GID:-1000}"', compose)
        self.assertIn(
            "paddle_models:/home/app/.paddlex/official_models", compose
        )
        self.assertNotIn("/root/.paddlex", compose)

    def test_backend_direct_dependencies_have_a_lock_source(self):
        project_dir = Path(__file__).resolve().parents[1]
        dockerfile = (project_dir / "Dockerfile").read_text(encoding="utf-8")
        direct_requirements = (project_dir / "requirements.in").read_text(encoding="utf-8")

        self.assertIn("COPY requirements.lock .", dockerfile)
        self.assertFalse((project_dir / "uv.lock").exists())
        for dependency in (
            "bcrypt==",
            "cryptography==",
            "fastapi==",
            "chromadb==",
            "langchain-openai==",
            "paddleocr==",
            "paddlepaddle==",
            "python-docx==",
        ):
            self.assertIn(dependency, direct_requirements)

    def test_backend_receives_the_model_key_encryption_settings(self):
        project_dir = Path(__file__).resolve().parents[1]
        compose = (project_dir / "docker-compose.yml").read_text(encoding="utf-8")
        env_example = (project_dir / ".env.example").read_text(encoding="utf-8")

        for variable in (
            "MODEL_API_KEY_ENCRYPTION_KEY",
            "MODEL_API_KEY_PREVIOUS_ENCRYPTION_KEYS",
        ):
            self.assertIn(variable, compose)
            self.assertIn(variable, env_example)

    def test_retrieval_planning_is_opt_in_and_documented_in_the_env_example(self):
        project_dir = Path(__file__).resolve().parents[1]
        compose = (project_dir / "docker-compose.yml").read_text(encoding="utf-8")
        env_example = (project_dir / ".env.example").read_text(encoding="utf-8")

        self.assertIn("CHAT_RETRIEVAL_PLANNING: ${CHAT_RETRIEVAL_PLANNING:-false}", compose)
        self.assertIn("CHAT_RETRIEVAL_PLANNING=false", env_example)

    def test_query_variant_is_opt_in_and_documented_in_the_env_example(self):
        project_dir = Path(__file__).resolve().parents[1]
        compose = (project_dir / "docker-compose.yml").read_text(encoding="utf-8")
        env_example = (project_dir / ".env.example").read_text(encoding="utf-8")

        self.assertIn("CHAT_QUERY_VARIANT: ${CHAT_QUERY_VARIANT:-false}", compose)
        self.assertIn("CHAT_QUERY_VARIANT=false", env_example)

    def test_chat_proxy_disables_response_buffering(self):
        frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
        dockerfile = (frontend_dir / "Dockerfile").read_text(encoding="utf-8")
        nginx_config = (frontend_dir / "nginx.conf").read_text(encoding="utf-8")
        chat_location = nginx_config.split("location = /api/chat {", 1)[1].split("}", 1)[0]
        self.assertIn("COPY nginx.conf /etc/nginx/conf.d/default.conf", dockerfile)
        self.assertIn("proxy_pass http://backend:8000/api/chat;", chat_location)
        self.assertIn("proxy_http_version 1.1;", chat_location)
        self.assertIn('proxy_set_header Connection "";', chat_location)
        self.assertIn("proxy_buffering off;", chat_location)
        self.assertIn("proxy_cache off;", chat_location)
        self.assertIn("proxy_read_timeout 300s;", chat_location)
        self.assertIn("client_max_body_size 52M;", nginx_config)

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
        project_dir = Path(__file__).resolve().parents[1]
        check_script = (
            project_dir / "scripts" / "check.sh"
        ).read_text(encoding="utf-8")
        compose = (project_dir / "docker-compose.yml").read_text(encoding="utf-8")
        dockerfile = (project_dir / "Dockerfile").read_text(encoding="utf-8")
        for expected in (
            'run_step_with_retries "Docker images" 3',
            "docker compose build backend frontend backend-check frontend-check e2e",
            "docker compose run --rm --no-deps backend-check",
            "PYTHONPYCACHEPREFIX=/tmp/pymodel-pycache",
            "python -m unittest discover -s tests -v",
            "python scripts/openapi_snapshot.py --check",
            "docker compose run --rm --no-deps frontend-check",
            "docker compose run --rm --no-deps e2e",
            "frontend nginx -t",
        ):
            self.assertIn(expected, check_script)
        self.assertNotIn("uvx", check_script)
        self.assertIn('target: check', compose)
        self.assertIn(
            "ruff check api.py app_storage.py routers services tests",
            compose,
        )
        self.assertIn(
            "python -m compileall -q api.py app_storage.py routers services tests",
            compose,
        )
        self.assertIn("FROM runtime AS check", dockerfile)
        self.assertIn("ruff==0.16.0", dockerfile)
        self.assertIn('"pyright[nodejs]==1.1.408"', dockerfile)
        self.assertIn("uv==0.11.18", dockerfile)
        self.assertIn("mkdir -p /home/app/.paddlex", dockerfile)
        self.assertIn("chown -R app:app /home/app/.paddlex", dockerfile)
        self.assertIn("RUFF_CACHE_DIR: /tmp/ruff-cache", compose)
        self.assertIn("UV_CACHE_DIR: /tmp/uv-cache", compose)

        smoke_script = (
            Path(__file__).resolve().parents[1] / "scripts" / "smoke-test.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("docker compose run --rm --no-deps", smoke_script)
        self.assertIn("python scripts/deployment_smoke.py", smoke_script)
        self.assertIn("--base-url http://backend:8000", smoke_script)
        self.assertNotIn("python3 scripts/deployment_smoke.py", smoke_script)
        self.assertIn("docker compose restart backend frontend", smoke_script)

        e2e_dockerfile = (
            Path(__file__).resolve().parents[1] / "frontend" / "Dockerfile.e2e"
        ).read_text(encoding="utf-8")
        self.assertIn("mcr.microsoft.com/playwright:v1.55.1-noble", e2e_dockerfile)
        self.assertIn('CMD ["npm", "run", "test:e2e"]', e2e_dockerfile)

    def test_cleanup_and_source_archive_preserve_runtime_data(self):
        project_dir = Path(__file__).resolve().parents[1]
        cleanup = (project_dir / "scripts" / "clean-workspace.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("frontend/node_modules", cleanup)
        self.assertIn(".uv-cache", cleanup)
        self.assertNotIn('rm -rf -- "$PROJECT_DIR/storage"', cleanup)

    def test_pyright_starts_with_typed_boundary_modules(self):
        project_dir = Path(__file__).resolve().parents[1]
        pyproject = (project_dir / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn("[tool.pyright]", pyproject)
        self.assertIn('pythonVersion = "3.10"', pyproject)
        self.assertIn('typeCheckingMode = "standard"', pyproject)
        for module in (
            "services/chat/contracts.py",
            "services/chat/rewrite.py",
            "services/chat/persistence_contracts.py",
            "services/chat/sessions.py",
            "services/chat/sources.py",
            "services/core/contracts.py",
            "services/core/llm_config.py",
            "services/core/errors.py",
            "services/core/settings.py",
            "services/documents/contracts.py",
            "services/documents/types.py",
            "services/indexing/contracts.py",
        ):
            self.assertIn(f'"{module}"', pyproject)
