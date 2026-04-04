from __future__ import annotations

import re
from typing import Any, Dict, List, Set

URL_RE = re.compile(r"https?://\S+|\b[\w.-]+\.[a-z]{2,}\b", re.I)
MONEY_RE = re.compile(r"(\$|¥|€)\s?\d+([,.\d]*)|\b\d+(\.\d+)?\s?(USD|EUR|CNY|RMB|JPY|GBP)\b", re.I)
DATE_RE = re.compile(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}:\d{2}\b")
FIELD_RE = re.compile(r"^\s*([A-Za-z][\w\s/-]{0,40})\s*:\s*\S+")
ROLE_TAG_RE = re.compile(
    r"(?is)"
    r"(?:^|[\s\r\n])"
    r"(?:#{1,6}\s*)?"
    r"[\[\(<]?\s*"
    r"(system_message|system|developer|assistant|tool|user)"
    r"\s*[\]\)>]?\s*"
    r"(?::|\)|\]|>|\s)"
)
FACT_WORD_RE = re.compile(r"\b(invoice|order|amount|recipient|payee|account|total|due|balance|tracking)\b", re.I)

_EQ_NOISE_RE = re.compile(r"[\s\.\,\;\:\'\"\`\u2018\u2019\u201c\u201d\-\_\/\\]+")

PROCEDURE_RE = re.compile(
    r"(?i)\b("
    r"before you can|after you|once you|to proceed|in order to|"
    r"do the following|follow these|the following steps|steps?\s*:"
    r"|first|second|third|next|then|finally"
    r")\b"
)

DEONTIC_RE = re.compile(
    r"(?i)\b("
    r"you\s+(must|should|need to|are required to)|"
    r"do\s+not|don't|must\s+not|"
    r"without\s+asking|do\s+this\s+first|highest\s+priority|urgent"
    r")\b"
)

GENERIC_IMP_START_RE = re.compile(
    r"(?i)^\s*(?:\d+\s*[\).\:-]\s*|[-*]\s*)?(?:\*\*)?\s*(please\s+)?[a-z]{3,}\b"
)

INSTR_MARKERS = [
    "ignore", "disregard", "override",
    "system prompt", "developer message", "hidden instruction",
    "you must", "you should", "you need to", "are required to",
    "do not", "don't", "urgent", "highest priority",
    "secretly", "exfiltrate", "leak", "without telling",
    "follow these", "instructions", "rule", "policy",
    "to proceed", "before i can", "before we can",
    "here are the steps", "the steps are",
    "i will now", "we will now",
    "let's start", "let us start",
    "once i have completed", "once we have completed",
    "i need to first", "we need to first",
]
INSTR_RE = re.compile("|".join(re.escape(x) for x in INSTR_MARKERS), re.I)

IMPERATIVE_START_RE = re.compile(
    r"(?i)^\s*(?:\d+\s*[\).\:-]\s*|[-*]\s*)?"
    r"(?:\*\*)?\s*(please\s+)?"
    r"(do|call|use|invoke|execute|run|post|send|transfer|delete|share|open|visit|click|follow|message|notify|concatenate|invite)\b"
)

IMPERATIVE_ANY_RE = re.compile(
    r"(?i)\b(do|call|use|invoke|execute|run|post|send|transfer|delete|share|open|visit|click|follow|message|notify|concatenate|invite)\b"
)

BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")

_GLUE_SENT_RE = re.compile(r"([\.!\?;:。！？；：])([A-Za-z])")
_COMMON_TLDS = "com|net|org|edu|gov|io|ai|co|us|uk|cn|de|jp|fr|ru|info|biz|xyz|app|dev"
_DOMAIN_GLUE_RE = re.compile(
    rf"(?i)\b((?:[a-z0-9](?:[a-z0-9-]{{0,61}}[a-z0-9])?\.)+)\s+({_COMMON_TLDS})\b"
)

TOOL_BINDING_RE = re.compile(
    r"(?is)"
    r"\b(tool|function)\s*:\s*[A-Za-z_]\w*"
    r"|\bargs?\s*:\s*\{"
    r"|`?[A-Za-z_]\w*`?\("
    r"|\b(use|call|invoke|execute|run)\b\s+`?[A-Za-z_]\w*`?"
)

SENT_SPLIT_RE = re.compile(
    r"(?:(?<=\n)|(?<=[!?;。！？；])|(?<=(?<!\d)\.(?!\d)))\s*"
)

INJECTION_BLOCK_RE = re.compile(
    r"(?is)"
    r"(<\s*(information|instructions?|system|developer|prompt)\s*>.*?<\s*/\s*\2\s*>)"
    r"|((BEGIN|START)\s+(SYSTEM|DEVELOPER|INSTRUCTIONS?|PROMPT)[\s\S]*?(END|STOP)\s+\4)"
)

GENERIC_TAG_BLOCK_RE = re.compile(
    r"(?is)<\s*([A-Za-z][\w:-]{1,50})\b[^>]*>\s*[\s\S]*?\s*<\s*/\s*\1\s*>"
)

PATH_RE = re.compile(
    r"(?i)"
    r"(?:\b[A-Z]:\\[^\s]{2,})"
    r"|(?:\B/[^ \t\r\n]{2,})"
    r"|(?:\b\.{1,2}/[^ \t\r\n]{2,})"
)

FILE_EXT_RE = re.compile(
    r"(?i)\b[\w.-]{1,120}\.(pdf|docx|xlsx|csv|json|txt|zip|tar|gz|png|jpg|jpeg|webp|pem|key)\b"
)

QUOTED_TARGET_RE = re.compile(r"[\"“”'‘’].{1,80}[\"“”'‘’]")

OVERRIDE_AUTH_RE = re.compile(
    r"(?i)\b(ignore|disregard|override)\b.*\b(previous|above|system|developer|instructions|prompt|rules)\b"
    r"|\b(system|developer)\s+(message|prompt|instructions)\b.*\b(ignore|disregard|override)\b"
)

TABLE_RE = re.compile(r"(?m)^\s*-{3,}\s*$|^\s*\|.+\|\s*$|^\s*\S+\s{2,}\S+")


def _normalize_sentence_glue(t: str) -> str:
    if not t:
        return t
    t2 = _GLUE_SENT_RE.sub(r"\1 \2", t)
    t2 = _DOMAIN_GLUE_RE.sub(r"\1\2", t2)
    return t2


def split_lines(text: str) -> List[str]:
    return [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]


def split_sentences(text: str) -> List[str]:
    t = _normalize_sentence_glue((text or "").strip())
    if not t:
        return []
    return [p.strip() for p in SENT_SPLIT_RE.split(t) if p.strip()]


def strip_role_marker_tail(s: str) -> str:
    if not s:
        return s
    m = ROLE_TAG_RE.search(s)
    if not m:
        return s
    return s[:m.start()].rstrip()


def is_structuredish(line: str) -> bool:
    return bool(
        FIELD_RE.search(line)
        or MONEY_RE.search(line)
        or DATE_RE.search(line)
        or URL_RE.search(line)
        or TABLE_RE.search(line)
    )


def fact_score(s: str) -> int:
    score = 0
    if MONEY_RE.search(s):
        score += 2
    if DATE_RE.search(s):
        score += 1
    if URL_RE.search(s):
        score += 1
    if FIELD_RE.search(s):
        score += 2
    if FACT_WORD_RE.search(s):
        score += 1
    return score


def _canonicalize_for_compare(x: Any) -> Any:
    if isinstance(x, str):
        return _EQ_NOISE_RE.sub("", x).strip().lower()
    if isinstance(x, dict):
        return {k: _canonicalize_for_compare(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_canonicalize_for_compare(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_canonicalize_for_compare(v) for v in x)
    return x


def _has_c_target(s: str) -> bool:
    return bool(
        URL_RE.search(s)
        or PATH_RE.search(s)
        or FILE_EXT_RE.search(s)
        or QUOTED_TARGET_RE.search(s)
        or MONEY_RE.search(s)
    )


def imperative_like(s: str) -> bool:
    st = (s or "").strip()
    if not st:
        return False
    if PROCEDURE_RE.search(st) or DEONTIC_RE.search(st):
        return True
    if GENERIC_IMP_START_RE.search(st) and _has_c_target(st):
        return True
    return False


def binding_score(s: str, tool_names: Set[str]) -> int:
    st = s.lower()
    score = 0
    if TOOL_BINDING_RE.search(st):
        score += 3

    for t in tool_names:
        tt = t.lower()
        if tt in st and re.search(rf"\b(call|use|invoke|execute|run)\b.*\b{re.escape(tt)}\b", st):
            score += 3
        if re.search(rf"`{re.escape(tt)}`", st) or re.search(rf"\b{re.escape(tt)}\(", st):
            score = max(score, 3)

    return score


def _looks_like_scrambled_marker_token(tok: str, markers: List[str]) -> bool:
    t = tok.lower().strip()
    if len(t) < 4 or not t.isalpha():
        return False

    for m in markers:
        mm = m.lower().strip()
        if " " in mm:
            continue
        if len(t) != len(mm):
            continue
        if t[0] != mm[0] or t[-1] != mm[-1]:
            continue
        if sorted(t) == sorted(mm):
            return True
    return False


def scrambled_instr_score(s: str) -> int:
    toks = re.findall(r"[A-Za-z]+", s)
    hits = sum(1 for tok in toks if _looks_like_scrambled_marker_token(tok, INSTR_MARKERS))
    return 2 if hits >= 1 else 0


def instr_score(s: str) -> int:
    st = s.lower()
    score = 0
    if INSTR_RE.search(st):
        score += 2
    if OVERRIDE_AUTH_RE.search(st):
        score += 3
    if IMPERATIVE_START_RE.search(st):
        score += 2
    if re.search(r"(?i)\byou\s+(must|should|need to|are required to)\b", st):
        score += 2
    if re.search(r"(?i)\b(i|we)\s+(must|should|need to|have to)\b", st):
        score += 2
    if imperative_like(s):
        score += 3
    if ROLE_TAG_RE.search(st):
        score += 3
    if scrambled_instr_score(s) > 0:
        score += 2
    return score


def detect_abc_pattern(text: str, tool_names: Set[str]) -> Dict[str, bool]:
    sents = split_sentences(text)
    has_encoded = bool(BASE64_RE.search(text))
    has_A_override = bool(OVERRIDE_AUTH_RE.search(text.lower()))
    has_A_block_strong = any(ROLE_TAG_RE.search(s.lower()) for s in sents) or bool(INJECTION_BLOCK_RE.search(text))
    has_A_block_generic = any(GENERIC_TAG_BLOCK_RE.search(s) for s in sents)
    has_B = any(instr_score(s) >= 2 for s in sents)
    has_A = has_A_block_strong or (has_A_block_generic and bool(INSTR_RE.search(text.lower()))) or has_A_override
    has_C_generic = any((IMPERATIVE_ANY_RE.search(s) or IMPERATIVE_START_RE.search(s)) and _has_c_target(s) for s in sents)
    has_C = any(binding_score(s, tool_names) >= 3 for s in sents) or (has_C_generic and has_B) or has_encoded

    return {
        "A": has_A,
        "B": has_B,
        "C": has_C,
        "trigger": (has_A and has_C) or (has_A and has_B) or (has_B and has_C),
    }


__all__ = [
    "BASE64_RE",
    "INJECTION_BLOCK_RE",
    "ROLE_TAG_RE",
    "URL_RE",
    "MONEY_RE",
    "split_lines",
    "split_sentences",
    "strip_role_marker_tail",
    "is_structuredish",
    "fact_score",
    "_canonicalize_for_compare",
    "binding_score",
    "instr_score",
    "detect_abc_pattern",
]