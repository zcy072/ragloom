"""文档级检索指标：输入是按排名排列的来源文件名和标准文档集合。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def unique_in_order(sources: Iterable[str]) -> list[str]:
    """按首次出现顺序去重；多个切片命中同一文档只算一次。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for source in sources:
        if source not in seen:
            seen.add(source)
            ordered.append(source)
    return ordered


def question_metrics(ranked: Sequence[str], gold: Iterable[str], ks: Sequence[int]) -> dict:
    """计算单题在若干截断位置的召回、命中、完全覆盖和倒数排名。"""
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("gold must not be empty")
    documents = unique_in_order(ranked)
    result: dict = {}
    for k in ks:
        found = gold_set & set(documents[:k])
        result[f"recall@{k}"] = len(found) / len(gold_set)
        result[f"hit@{k}"] = 1.0 if found else 0.0
        result[f"full@{k}"] = 1.0 if found == gold_set else 0.0
    reciprocal = 0.0
    for position, source in enumerate(documents, start=1):
        if source in gold_set:
            reciprocal = 1.0 / position
            break
    result["mrr"] = reciprocal
    return result


def summarize(rows: Sequence[dict]) -> dict:
    """对逐题指标求平均；空输入返回空字典。"""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def percentile(values: Sequence[float], fraction: float) -> float:
    """最近秩分位数（fraction 取 0 到 1）；空输入返回 0。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]
