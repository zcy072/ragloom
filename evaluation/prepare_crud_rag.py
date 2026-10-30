#!/usr/bin/env python3
"""把 CRUD-RAG 抽样成评测用的小知识库：语料文件 + 带标准文档的问题集。

只使用标准库。用法见 evaluation/README.md。
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import random
import subprocess
from pathlib import Path

DOC_TASKS = ("questanswer_2docs", "questanswer_3docs")
SNIPPET_CHARS = 30


def load_corpus(repo: Path) -> list[str]:
    """读取全部新闻（每个非空行一篇），去除完全重复的行。"""
    seen: set[str] = set()
    lines: list[str] = []
    for path in sorted(glob.glob(str(repo / "data" / "80000_docs" / "*"))):
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.rstrip("\n")
                if line.strip() and line not in seen:
                    seen.add(line)
                    lines.append(line)
    return lines


def body_of(text: str) -> str:
    """去掉标题行前缀，只保留“正文：”之后的内容。"""
    return text.split("正文：", 1)[-1].strip()


class CorpusLocator:
    """用正文中段片段在整个语料里定位一篇新闻所在行。"""

    def __init__(self, lines: list[str]) -> None:
        self._joined = "\n".join(lines)
        self._starts: list[int] = []
        offset = 0
        for line in lines:
            self._starts.append(offset)
            offset += len(line) + 1

    def locate(self, news_text: str) -> int | None:
        body = body_of(news_text)
        if len(body) < SNIPPET_CHARS * 2:
            return None
        middle = len(body) // 2
        snippet = body[middle : middle + SNIPPET_CHARS]
        position = self._joined.find(snippet)
        if position < 0 or self._joined.find(snippet, position + 1) >= 0:
            return None  # 找不到或片段出现多次，视为无法唯一定位
        return bisect.bisect_right(self._starts, position) - 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="CRUD_RAG 仓库目录")
    parser.add_argument("--output", type=Path, required=True, help="输出目录（应为空目录）")
    parser.add_argument("--questions", type=int, default=200)
    parser.add_argument("--distractors", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extension", choices=("txt", "md"), default="txt",
        help="语料文件扩展名；md 会让固定和结构化两条路线同时入库",
    )
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"输出目录非空：{args.output}")
    corpus_dir = args.output / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    lines = load_corpus(args.repo)
    locator = CorpusLocator(lines)
    qa = json.loads((args.repo / "data" / "crud_split" / "split_merged.json").read_text("utf-8"))

    candidates = []
    for task in DOC_TASKS:
        news_keys = (
            ("news1", "news2") if task == "questanswer_2docs" else ("news1", "news2", "news3")
        )
        for entry in qa[task]:
            positions = [locator.locate(entry[key]) for key in news_keys]
            if all(p is not None for p in positions) and len(set(positions)) == len(positions):
                candidates.append((task, entry, positions))

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    chosen = []
    used_questions: set[str] = set()
    for task, entry, positions in candidates:
        if entry["questions"] in used_questions:
            continue
        used_questions.add(entry["questions"])
        chosen.append((task, entry, positions))
        if len(chosen) == args.questions:
            break

    gold_lines = {p for _task, _entry, positions in chosen for p in positions}
    others = [i for i in range(len(lines)) if i not in gold_lines]
    distractor_lines = set(rng.sample(others, min(args.distractors, len(others))))
    selected = sorted(gold_lines | distractor_lines)
    names = {line_index: f"news_{line_index:06d}.{args.extension}" for line_index in selected}
    for line_index, name in names.items():
        (corpus_dir / name).write_text(lines[line_index], encoding="utf-8")

    with open(args.output / "questions.jsonl", "w", encoding="utf-8") as handle:
        for number, (task, entry, positions) in enumerate(chosen, start=1):
            row = {
                "id": f"q{number:04d}",
                "task": task,
                "question": entry["questions"],
                "gold": [names[p] for p in positions],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        revision = subprocess.run(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = ""
    manifest = {
        "dataset": "CRUD-RAG (IAAR-Shanghai/CRUD_RAG)",
        "dataset_revision": revision,
        "seed": args.seed,
        "tasks": list(DOC_TASKS),
        "corpus_lines_unique": len(lines),
        "usable_questions": len(candidates),
        "questions": len(chosen),
        "gold_documents": len(gold_lines),
        "distractor_documents": len(distractor_lines),
        "corpus_documents": len(selected),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
