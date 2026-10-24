"""Dependency-free heading patterns shared by document parsing and chunking."""

import re

CHAPTER_HEADING = re.compile(r"^第[一二三四五六七八九十百千万零〇0-9]+[章节篇部分]")
STANDARD_HEADING = re.compile(r"^(\d+(?:\.\d+)+)\s+")
STANDARD_TOP_HEADING = re.compile(r"^\d+\s+")
CHINESE_MAJOR_HEADING = re.compile(r"^[一二三四五六七八九十百千万零〇]+[、.]")
CHINESE_SUB_HEADING = re.compile(r"^[（(][一二三四五六七八九十百千万零〇]+[）)]")
ARABIC_ITEM_HEADING = re.compile(r"^\d+[.、]\s*")
PAREN_NUMBER_HEADING = re.compile(r"^[（(]\d+[）)]")
CIRCLED_NUMBER_HEADING = re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]")
APPENDIX_HEADING = re.compile(
    r"^附录(?:\s|[A-ZＡ-Ｚ0-9０-９]|[一二三四五六七八九十])"
)
APPENDIX_CHILD_HEADING = re.compile(r"^[A-ZＡ-Ｚ]\s*\.\s*\d+")
PREFACE_HEADING = re.compile(r"^(?:前言|摘要|abstract|preface|foreword)$", re.IGNORECASE)
