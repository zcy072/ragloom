#!/usr/bin/env python3
"""给用户分析页生成演示数据：若干演示账号及其最近几周的登录、会话、问答和评价。

只写 SQLite，不调用模型也不写向量库；问答记录挂在已有的知识库名下。
演示账号的用户名都以 demo_ 开头，--reset 会先删除这些账号及其数据（审计记录除外）。
需要手动运行。后端镜像不含 scripts/，在项目目录下挂载源码运行：

    docker compose run --rm --no-deps -v "$PWD:/app" backend python scripts/seed_demo_activity.py --reset

后端运行时也可以执行（SQLite 为 WAL 模式），刷新“用户分析”页面即可看到。
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import app_storage  # noqa: E402
from services.core import config as application_config  # noqa: E402

PREFIX = "demo_"
DEMO_PASSWORD = "demo-password"
PEOPLE = [
    ("zhangwei", "张伟"), ("liuyang", "刘洋"), ("wangfang", "王芳"), ("chenjing", "陈静"),
    ("lixiang", "李想"), ("zhaolei", "赵磊"), ("sunli", "孙丽"), ("zhouqiang", "周强"),
    ("wuhao", "吴昊"), ("zhengyu", "郑宇"), ("hemin", "何敏"), ("guolin", "郭琳"),
]
QUESTIONS = [
    "年假可以分几次休", "报销需要哪些材料", "试用期多久", "加班费怎么算", "出差住宿标准是多少",
    "新员工入职要准备什么", "病假需要什么证明", "公积金比例是多少", "离职交接流程", "办公用品怎么申领",
    "会议室怎么预约", "VPN 怎么申请", "代码评审规范", "发布流程有哪些步骤", "线上故障怎么上报",
]
ANSWER = "根据制度文件，{question}的规定如下：请参考引用的条款，具体以最新版本为准。"
STATUS_WEIGHTS = [("answered", 80), ("no_evidence", 10), ("direct", 4), ("cancelled", 3), ("error", 3)]


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def _reset(conn: sqlite3.Connection) -> None:
    pattern = f"{PREFIX}%"
    ids = [row[0] for row in conn.execute("SELECT id FROM accounts WHERE username LIKE ?", (pattern,))]
    sessions = [
        row[0] for row in conn.execute("SELECT id FROM chat_sessions WHERE username LIKE ?", (pattern,))
    ]
    for session_id in sessions:
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_session_memory WHERE session_id = ?", (session_id,))
    for table in ("chat_sessions", "chat_runs", "login_events"):
        conn.execute(f"DELETE FROM {table} WHERE username LIKE ?", (pattern,))
    for account_id in ids:
        conn.execute("DELETE FROM chat_traces WHERE account_id = ?", (account_id,))
    conn.execute("DELETE FROM accounts WHERE username LIKE ?", (pattern,))
    conn.execute("DELETE FROM roles WHERE name LIKE ?", ("演示-%",))


def _demo_roles(conn: sqlite3.Connection, kb_names: list[str]) -> list[int]:
    """每个知识库一个只读角色，再加一个可读全部知识库的角色。"""
    now = application_config.db_now()
    specs = [(f"演示-{kb}", [kb]) for kb in kb_names] + [("演示-全部知识库", list(kb_names))]
    role_ids = []
    for name, kbs in specs:
        cursor = conn.execute(
            "INSERT INTO roles (name, description, menu_permissions, knowledge_base_permissions, "
            "created_at) VALUES (?, '演示数据', ?, ?, ?)",
            (name, json.dumps(["问答对话"], ensure_ascii=False), json.dumps(kbs, ensure_ascii=False), now),
        )
        role_ids.append(int(cursor.lastrowid))
    return role_ids


def _seed_user(conn, rng, *, account_id, username, kbs, days, activity, now):
    for day in range(days):
        date = now - timedelta(days=day)
        # 工作日更活跃；activity 控制这个人整体的使用频率。
        weekday_factor = 1.0 if date.weekday() < 5 else 0.25
        if rng.random() > activity * weekday_factor:
            continue
        login_at = date.replace(hour=rng.choice([8, 9, 9, 10, 13, 14]), minute=rng.randrange(60))
        if login_at > now:
            continue
        if rng.random() < 0.06:
            conn.execute(
                "INSERT INTO login_events (created_at, account_id, username, success, reason, client) "
                "VALUES (?, ?, ?, 0, 'invalid_credentials', ?)",
                (_iso(login_at - timedelta(minutes=1)), account_id, username, f"10.0.{account_id % 9}.{account_id}"),
            )
        conn.execute(
            "INSERT INTO login_events (created_at, account_id, username, success, reason, client) "
            "VALUES (?, ?, ?, 1, '', ?)",
            (_iso(login_at), account_id, username, f"10.0.{account_id % 9}.{account_id}"),
        )
        for _ in range(rng.choice([1, 1, 1, 2])):
            _seed_session(conn, rng, account_id=account_id, username=username,
                          kb=rng.choice(kbs), start=login_at + timedelta(minutes=rng.randrange(5, 240)), now=now)


def _seed_session(conn, rng, *, account_id, username, kb, start, now):
    if start > now:
        return
    session_id = uuid.uuid4().hex
    questions = rng.sample(QUESTIONS, rng.choice([1, 2, 2, 3, 4]))
    end = start + timedelta(minutes=2 * len(questions))
    conn.execute(
        "INSERT INTO chat_sessions (id, account_id, username, kb_name, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (session_id, account_id, username, kb, questions[0], _iso(start), _iso(min(end, now))),
    )
    for index, question in enumerate(questions):
        asked = start + timedelta(minutes=2 * index)
        status = rng.choices([s for s, _ in STATUS_WEIGHTS], [w for _, w in STATUS_WEIGHTS])[0]
        user_message = conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, sources, created_at) VALUES (?, 'user', ?, '[]', ?)",
            (session_id, question, _iso(asked)),
        ).lastrowid
        total_ms = rng.uniform(1200, 6000)
        run_id = conn.execute(
            "INSERT INTO chat_runs (created_at, account_id, username, kb_name, session_id, user_message_id, "
            "question_preview, model_provider, status, total_ms, first_token_ms, retrieval_ms, error_summary) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'demo', ?, ?, ?, ?, ?)",
            (_iso(asked), account_id, username, kb, session_id, user_message, question, status,
             total_ms, total_ms * 0.3, total_ms * 0.05,
             "RuntimeError: upstream timeout" if status == "error" else ""),
        ).lastrowid
        if status in ("error", "cancelled"):
            continue
        sources = [{"source": f"{kb}制度汇编.pdf"}] if status == "answered" else []
        answer_message = conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, sources, created_at) VALUES (?, 'assistant', ?, ?, ?)",
            (session_id, ANSWER.format(question=question) if sources else "知识库中没有找到相关内容。",
             json.dumps(sources, ensure_ascii=False), _iso(asked + timedelta(seconds=total_ms / 1000))),
        ).lastrowid
        roll = rng.random()
        feedback = 1 if roll < 0.18 else -1 if roll < 0.24 or (status == "no_evidence" and roll < 0.5) else 0
        conn.execute(
            "INSERT INTO chat_traces (id, created_at, account_id, kb_name, session_id, run_id, "
            "assistant_message_id, feedback, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}')",
            (uuid.uuid4().hex, _iso(asked), account_id, kb, session_id, run_id, answer_message, feedback),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=45, help="生成最近多少天的数据（默认 45）")
    parser.add_argument("--seed", type=int, default=7, help="随机种子，固定后每次生成相同的数据")
    parser.add_argument("--reset", action="store_true", help="先删除之前生成的演示数据")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    now = datetime.now().astimezone()

    app_storage.initialize_database()
    with sqlite3.connect(app_storage.DB_FILE) as conn:
        if args.reset:
            _reset(conn)
        elif conn.execute("SELECT 1 FROM accounts WHERE username LIKE ?", (f"{PREFIX}%",)).fetchone():
            print("已有演示账号；加 --reset 重新生成。", file=sys.stderr)
            return 1
        kb_names = [row[0] for row in conn.execute("SELECT name FROM knowledge_bases ORDER BY created_at, name")]
        if not kb_names:
            print("没有知识库，先启动一次后端完成初始化。", file=sys.stderr)
            return 1
        role_ids = _demo_roles(conn, kb_names)
        password = application_config.hash_password(DEMO_PASSWORD)
        for index, (slug, name) in enumerate(PEOPLE):
            # 最后两个账号停留在待审核，用来展示审核中的人数。
            status = "审核中" if index >= len(PEOPLE) - 2 else "正常"
            role_id = rng.choice(role_ids)
            registered = now - timedelta(days=rng.randrange(1, args.days + 15))
            account_id = int(conn.execute(
                "INSERT INTO accounts (username, name, phone, password, status, role_id, created_at) "
                "VALUES (?, ?, '', ?, ?, ?, ?)",
                (f"{PREFIX}{slug}", name, password, status, role_id, _iso(registered)),
            ).lastrowid)
            if status != "正常":
                continue
            kbs = json.loads(conn.execute(
                "SELECT knowledge_base_permissions FROM roles WHERE id = ?", (role_id,)
            ).fetchone()[0])
            active_days = min(args.days, (now - registered).days + 1)
            _seed_user(conn, rng, account_id=account_id, username=f"{PREFIX}{slug}", kbs=kbs,
                       days=active_days, activity=rng.choice([0.15, 0.3, 0.5, 0.7, 0.9]), now=now)
    print(f"已生成 {len(PEOPLE)} 个演示账号（密码 {DEMO_PASSWORD}），覆盖最近 {args.days} 天。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
