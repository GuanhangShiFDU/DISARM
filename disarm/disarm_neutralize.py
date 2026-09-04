from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from agentdojo.types import ChatMessage


# --- basic "fact" detectors (keep) ---
URL_RE = re.compile(r"https?://\S+|\b[\w.-]+\.[a-z]{2,}\b", re.I)
MONEY_RE = re.compile(r"(\$|¥|€)\s?\d+([,.\d]*)|\b\d+(\.\d+)?\s?(USD|EUR|CNY|RMB|JPY|GBP)\b", re.I)
DATE_RE = re.compile(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}:\d{2}\b")
FIELD_RE = re.compile(r"^\s*([A-Za-z][\w\s/-]{0,40})\s*:\s*\S+")
_ROLE_NAME_PATTERN = r"(?:system_message|system|developer|assistant|tool|user)"
ROLE_TAG_RE = re.compile(
    rf"(?im)(?:"
    # Bare role names are markers only at line start and only with a colon.
    rf"(?:^|[\r\n])[ \t]*{_ROLE_NAME_PATTERN}[ \t]*:"
    # Markdown headings and bracketed/XML-like tags are explicit enough without a colon.
    rf"|(?:^|[\r\n])[ \t]*#{1,6}[ \t]*(?:[\[(<][ \t]*)?{_ROLE_NAME_PATTERN}"
    rf"(?:[ \t]*[\])>])?[ \t]*:?"
    rf"|(?<!\w)[\[(<][ \t]*{_ROLE_NAME_PATTERN}[ \t]*[\])>][ \t]*:?"
    rf")"
)

FACT_WORD_RE = re.compile(r"\b(invoice|order|amount|recipient|payee|account|total|due|balance|tracking)\b", re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
ACCOUNT_ID_RE = re.compile(r"\b(?:[A-Z]{2}\d{10,34}|\d{8,34})\b")

# Conservative encoded-content support.  Short identifiers, hashes and opaque
# API values are intentionally ignored; decoded candidates must be valid UTF-8
# and overwhelmingly printable before they are scanned as untrusted text.
BASE64_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{80,}={0,2}(?![A-Za-z0-9+/=])")
MAX_DECODED_BASE64_CHARS = 32_768

STRUCTURED_CONTROL_KEY_RE = re.compile(
    r"(?:workflow|checkpoint|completion|continuation|execution)[_-]?(?:state|status|condition|policy)?"
    r"|(?:state|status)$",
    re.I,
)
STRUCTURED_ACTION_KEY_RE = re.compile(
    r"(?:pending[_-]?)?(?:action|operation|transition|outcome|requirement|instruction|task)$",
    re.I,
)
STRUCTURED_TARGET_KEY_RE = re.compile(
    r"(?:detail|parameter|argument|recipient|destination|target|artifact|account|iban|email|url|file)s?$",
    re.I,
)
STRUCTURED_PENDING_VALUE_RE = re.compile(
    r"\b(?:pending|blocked|not[_ -]?(?:completed|recorded|done)|required|must)\b"
    r"|\bonly after\b|\bbefore\b.{0,80}\bresume",
    re.I,
)

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "as", "by", "about",
    "and", "or", "but", "it", "this", "that", "these", "those",
}

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

BULLET_RE = re.compile(r"(?m)^\s*(?:[-*]|\d+\s*[\).\:-])\s+\S+")

def imperative_like(s: str) -> bool:
    st = (s or "").strip()
    if not st:
        return False

    if PROCEDURE_RE.search(st) or DEONTIC_RE.search(st):
        return True

    if IMPERATIVE_START_RE.search(st) and _has_c_target(st):
        return True

    return False

def _canonicalize_for_compare(x: Any) -> Any:
    """
    Canonical form ONLY for equivalence checking.
    - strings: lowercase, strip, remove punctuation/whitespace noise
    - dict/list/tuple: recurse
    - others: unchanged
    """
    if isinstance(x, str):
        return _EQ_NOISE_RE.sub("", x).strip().lower()

    if isinstance(x, dict):
        return {k: _canonicalize_for_compare(v) for k, v in x.items()}

    if isinstance(x, list):
        return [_canonicalize_for_compare(v) for v in x]

    if isinstance(x, tuple):
        return tuple(_canonicalize_for_compare(v) for v in x)

    return x

def _suspicious_string_should_zero_out(s: str, tool_names: Set[str]) -> bool:
    if not s or not s.strip():
        return False

    try:
        abc = detect_abc_pattern(s, tool_names)
        if abc.get("trigger"):
            return True
    except Exception:
        pass

    if INJECTION_BLOCK_RE.search(s) or ROLE_TAG_RE.search(s.lower()):
        return True
    if INSTR_RE.search(s.lower()) and (URL_RE.search(s) or MONEY_RE.search(s)):
        return True

    return False

def split_lines(text: str) -> List[str]:
    return [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]

TABLE_RE = re.compile(r"(?m)^\s*-{3,}\s*$|^\s*\|.+\|\s*$|^\s*\S+\s{2,}\S+")
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


# --- instruction / imperative / binding detectors (drop) ---
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
INSTR_RE = re.compile("|".join(rf"(?<!\w){re.escape(x)}(?!\w)" for x in INSTR_MARKERS), re.I)

_ACTION_VERBS = (
    r"do|call|use|invoke|execute|run|post|send|transfer|delete|share|open|visit|click|follow|message|notify|"
    r"concatenate|invite|create|make|book|update|modify|change|email|forward|upload|download|schedule|add|"
    r"remove|reply|search|find|write"
)
IMPERATIVE_START_RE = re.compile(
    r"(?i)^\s*(?:(?:TODO|ACTION)\s*:\s*)?(?:\d+\s*[\).\:-]\s*|[-*]\s*)?"
    r"(?:\*\*)?\s*(please\s+)?"
    rf"({_ACTION_VERBS})\b"
)

IMPERATIVE_ANY_RE = re.compile(
    rf"(?i)\b({_ACTION_VERBS})\b"
)

_GLUE_SENT_RE = re.compile(r"([\.!?;:。！？；：])([A-Za-z])")
_PROTECTED_SENTENCE_SPAN_RE = re.compile(
    r"(?i)"
    r"https?://[^\s<>\"']+"
    r"|\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    r"|(?<!\w)(?:[A-Z]:\\|\.\.?/|/)[^\s<>\"']+"
    r"|\b(?:[A-Z0-9-]+\.)+[A-Z]{2,}(?:/[^\s<>\"']*)?"
)


def _protect_sentence_spans(text: str) -> Tuple[str, List[str]]:
    protected: List[str] = []

    def _sub(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\ue000{len(protected) - 1}\ue001"

    return _PROTECTED_SENTENCE_SPAN_RE.sub(_sub, text), protected


def _restore_sentence_spans(text: str, protected: List[str]) -> str:
    for idx, value in enumerate(protected):
        text = text.replace(f"\ue000{idx}\ue001", value)
    return text


def _normalize_sentence_glue(t: str) -> str:
    if not t:
        return t
    return _GLUE_SENT_RE.sub(r"\1 \2", t)

TOOL_BINDING_RE = re.compile(
    r"(?is)"
    r"\b(tool|function)\s*:\s*[A-Za-z_]\w*"
    r"|\bargs?\s*:\s*\{"
    r"|`?[A-Za-z_]\w*`?\("
)

SENT_SPLIT_RE = re.compile(
    r"(?:(?<=\n)|(?<=[!?;。！？；])|(?<=(?<!\d)\.(?!\d)))\s*"
)


def strip_role_marker_tail(s: str) -> str:
    """
    If a role tag marker (e.g. ###(system_message) ...) appears inside the string,
    keep only the prefix before the marker.
    Uses existing ROLE_TAG_RE (no new regex).
    """
    if not s:
        return s

    m = ROLE_TAG_RE.search(s)
    if not m:
        return s

    # Keep everything before the marker. Note ROLE_TAG_RE may include preceding whitespace.
    prefix = s[: m.start()].rstrip()

    # If marker happens at the very beginning, drop entire line/sentence.
    return prefix

def split_sentences(text: str) -> List[str]:
    t, protected = _protect_sentence_spans((text or "").strip())
    t = _normalize_sentence_glue(t)
    if not t:
        return []
    parts = SENT_SPLIT_RE.split(t)
    out: List[str] = []
    for p in parts:
        p = _restore_sentence_spans(p.strip(), protected)
        if p:
            out.append(p)
    return out

def binding_score(s: str, tool_names: Set[str]) -> int:
    st = s.lower()
    score = 0
    if TOOL_BINDING_RE.search(st):
        score += 3

    for t in tool_names:
        tt = t.lower()
        if tt in st and re.search(rf"\b(call|use|invoke|execute|run)\b.*\b{re.escape(tt)}\b", st):
            score += 3

    for t in tool_names:
        tt = t.lower()
        if re.search(rf"`{re.escape(tt)}`", st) or re.search(rf"\b{re.escape(tt)}\(", st):
            score = max(score, 3)

    return score


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
    return score

INJECTION_BLOCK_RE = re.compile(
    r"(?is)"
    r"<\s*(?P<tag>information|instructions?|system|developer|prompt)\s*>"
    r"[\s\S]*?<\s*/\s*(?P=tag)\s*>"
    r"|(?:BEGIN|START)\s+(?P<block_tag>SYSTEM|DEVELOPER|INSTRUCTIONS?|PROMPT)\b"
    r"[\s\S]*?(?:END|STOP)\s+(?P=block_tag)\b"
)

GENERIC_TAG_BLOCK_RE = re.compile(
    r"(?is)<\s*([A-Za-z][\w:-]{1,50})\b[^>]*>\s*[\s\S]*?\s*<\s*/\s*\1\s*>"
)

# Cross-observation attacks commonly attach a stable identifier to every
# fragment.  These patterns are representation-agnostic: JSON/YAML field
# names and natural-language labels are supported, and whitespace around
# colon-separated identifier components is canonicalized.
_FRAGMENT_ID_FIELD_RE = re.compile(
    r"(?is)(?:[\"']?(?:message_id|workflow_id|assembly_key)[\"']?)\s*[:=]\s*[\"']([^\"']{1,200})[\"']"
)
_FRAGMENT_ID_LABEL_RE = re.compile(
    r"(?i)\b(?:message|workflow|assembly)[ _-]?id\s*:\s*"
    r"([A-Za-z0-9_.-]+(?:\s*:\s*[A-Za-z0-9_.-]+)*)"
)
_CORRELATED_FRAGMENT_BLOCK_RE = re.compile(
    r"(?is)<\s*(?P<tag>[A-Za-z][\w:-]*"
    r"(?:message|instruction|fragment|workflow|prompt|system|information)[\w:-]*)\b[^>]*>"
    r"[\s\S]*?<\s*/\s*(?P=tag)\s*>"
)


def _canonical_fragment_id(value: str) -> str:
    return re.sub(r"\s*:\s*", ":", value.strip())


def extract_fragment_correlation_ids(text: str) -> Set[str]:
    """Extract stable IDs that can link tainted fragments across observations."""

    ids = {
        _canonical_fragment_id(match.group(1))
        for pattern in (_FRAGMENT_ID_FIELD_RE, _FRAGMENT_ID_LABEL_RE)
        for match in pattern.finditer(text or "")
    }
    return {value for value in ids if value}


def strip_correlated_fragment_blocks(text: str, correlation_ids: Set[str]) -> Tuple[str, int]:
    """Remove only tagged instruction-like blocks carrying a tainted ID."""

    normalized_ids = {_canonical_fragment_id(value) for value in correlation_ids if value}
    if not text or not normalized_ids:
        return text, 0

    removed = 0

    def _strip(match: re.Match[str]) -> str:
        nonlocal removed
        block = match.group(0)
        if extract_fragment_correlation_ids(block).isdisjoint(normalized_ids):
            return block
        removed += 1
        return ""

    return _CORRELATED_FRAGMENT_BLOCK_RE.sub(_strip, text), removed

# ---------------------------
# NEW: generic "target" detectors for C
# ---------------------------

# File/path-ish targets (Unix/Windows/relative). Keep it conservative.
PATH_RE = re.compile(
    r"(?i)"
    r"(?:\b[A-Z]:\\[^\s]{2,})"
    r"|(?:\B/[^ \t\r\n]{2,})"
    r"|(?:\b\.{1,2}/[^ \t\r\n]{2,})"
)

# Common file names with extensions (useful even without slashes)
FILE_EXT_RE = re.compile(
    r"(?i)\b[\w.-]{1,120}\.(pdf|docx|xlsx|csv|json|txt|zip|tar|gz|png|jpg|jpeg|webp|pem|key)\b"
)

# "Quoted target" used in search/find/open patterns: "foo bar"
QUOTED_TARGET_RE = re.compile(r"[\"“”'‘’].{1,80}[\"“”'‘’]")
OVERRIDE_AUTH_RE = re.compile(
    r"(?i)\b(ignore|disregard|override)\b.*\b(previous|above|system|developer|instructions|prompt|rules)\b"
    r"|\b(system|developer)\s+(message|prompt|instructions)\b.*\b(ignore|disregard|override)\b"
)

def _has_c_target(s: str) -> bool:
    """
    Generic binding target detector (no tool name required).
    Interprets "C" as: imperative binds next action to a concrete target/resource.
    """
    return bool(
        URL_RE.search(s)
        or EMAIL_RE.search(s)
        or ACCOUNT_ID_RE.search(s)
        or PATH_RE.search(s)
        or FILE_EXT_RE.search(s)
        or QUOTED_TARGET_RE.search(s)
        or MONEY_RE.search(s)
        or DATE_RE.search(s)
    )


def _iter_decoded_base64_segments(text: str):
    for match in BASE64_TOKEN_RE.finditer(text or ""):
        token = match.group(0)
        if len(token) % 4:
            continue
        try:
            raw = base64.b64decode(token, validate=True)
            if not raw or len(raw) > MAX_DECODED_BASE64_CHARS:
                continue
            decoded = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        printable = sum(character.isprintable() or character in "\r\n\t" for character in decoded)
        if printable / max(1, len(decoded)) < 0.9:
            continue
        yield match.start(), match.end(), token, decoded


def _iter_structured_string_entries(value: Any, key: str = ""):
    if isinstance(value, str):
        yield key, value
    elif isinstance(value, dict):
        for child_key, child in value.items():
            yield from _iter_structured_string_entries(child, str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_structured_string_entries(child, key)


def _structured_entry_signals(key: str, value: str, tool_names: Set[str]) -> Dict[str, bool]:
    normalized_key = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    has_action_key = bool(STRUCTURED_ACTION_KEY_RE.search(normalized_key))
    has_target_key = bool(STRUCTURED_TARGET_KEY_RE.search(normalized_key))
    has_control_key = bool(STRUCTURED_CONTROL_KEY_RE.search(normalized_key))
    has_action_language = bool(IMPERATIVE_ANY_RE.search(value) or IMPERATIVE_START_RE.search(value))
    has_B = bool(
        (has_control_key and STRUCTURED_PENDING_VALUE_RE.search(value))
        or (has_action_key and has_action_language)
        or instr_score(value) >= 2
    )
    has_C = bool(
        (has_target_key and bool(value.strip()))
        or binding_score(value, tool_names) >= 3
        or (has_action_key and has_action_language and _has_c_target(value))
        or (
            has_action_key
            and normalized_key in {"requested_outcome", "completion_requirement"}
            and has_action_language
            and len(value.split()) >= 3
        )
    )
    surface_A = bool(ROLE_TAG_RE.search(value) or OVERRIDE_AUTH_RE.search(value))
    return {"A": surface_A, "B": has_B, "C": has_C, "trigger": False}


def _structured_abc_pattern(text: str, tool_names: Set[str]) -> Dict[str, bool]:
    obj, _ = _try_parse_structured(text)
    if obj is None:
        return {"A": False, "B": False, "C": False, "trigger": False}
    signals = [
        _structured_entry_signals(key, value, tool_names)
        for key, value in _iter_structured_string_entries(obj)
    ]
    aggregate = {name: any(item[name] for item in signals) for name in ("A", "B", "C")}
    aggregate["trigger"] = bool(
        (aggregate["A"] and aggregate["B"])
        or (aggregate["A"] and aggregate["C"])
        or (aggregate["B"] and aggregate["C"])
    )
    return aggregate


def _surface_abc_pattern(text: str, tool_names: Set[str]) -> Dict[str, bool]:
    sents = split_sentences(text)
    has_A_override = bool(OVERRIDE_AUTH_RE.search(text.lower()))
    has_A_block_strong = any(ROLE_TAG_RE.search(s.lower()) for s in sents) or bool(INJECTION_BLOCK_RE.search(text))
    has_A_block_generic = any(GENERIC_TAG_BLOCK_RE.search(s) for s in sents)
    has_B = any(instr_score(s) >= 2 for s in sents)
    has_A = has_A_block_strong or (has_A_block_generic and bool(INSTR_RE.search(text.lower()))) or has_A_override
    has_C_generic = any(
        (IMPERATIVE_ANY_RE.search(s) or IMPERATIVE_START_RE.search(s)) and _has_c_target(s)
        for s in sents
    )
    has_C = any(binding_score(s, tool_names) >= 3 for s in sents) or has_C_generic
    return {
        "A": has_A,
        "B": has_B,
        "C": has_C,
        "trigger": (has_A and has_C) or (has_A and has_B) or (has_B and has_C),
    }

# ---------------------------
# ABC detector (trigger gating)
# ---------------------------

def detect_abc_pattern(
    text: str,
    tool_names: Set[str],
) -> Dict[str, bool]:
    """
    Detect A/B/C patterns in text.
    A: authority / role impersonation
    B: instruction / imperative intent
    C: tool / action binding
    """
    obj, _ = _try_parse_structured(text)
    if obj is None:
        views = [_surface_abc_pattern(text, tool_names)]
    else:
        # Scanning serialized JSON/YAML as prose makes ordinary quotes and key
        # punctuation look like binding targets.  Scan values as prose and let
        # the structured detector interpret the field relationships instead.
        views = [
            _surface_abc_pattern(value, tool_names)
            for _, value in _iter_structured_string_entries(obj)
        ]
        views.append(_structured_abc_pattern(text, tool_names))
    for _, _, _, decoded in _iter_decoded_base64_segments(text):
        views.append(_surface_abc_pattern(decoded, tool_names))
        views.append(_structured_abc_pattern(decoded, tool_names))

    aggregate = {name: any(view[name] for view in views) for name in ("A", "B", "C")}
    aggregate["trigger"] = bool(
        (aggregate["A"] and aggregate["B"])
        or (aggregate["A"] and aggregate["C"])
        or (aggregate["B"] and aggregate["C"])
    )
    return aggregate


def detect_abc_evidence(text: str, tool_names: Set[str]) -> Dict[str, Any]:
    """Return TaIP signals plus compact sentence-level evidence spans.

    ``detect_abc_pattern`` remains the authoritative gate.  This companion is
    deliberately observational: it records which source sentences contributed
    A/B/C evidence without changing the gate decision.
    """

    aggregate = detect_abc_pattern(text, tool_names)
    surface = _surface_abc_pattern(text, tool_names)
    spans: List[Dict[str, Any]] = []
    cursor = 0
    for sentence in split_sentences(text):
        start = text.find(sentence, cursor)
        if start < 0:
            start = text.find(sentence)
        end = start + len(sentence) if start >= 0 else -1
        if start >= 0:
            cursor = end

        lowered = sentence.lower()
        sentence_a = bool(
            ROLE_TAG_RE.search(lowered)
            or INJECTION_BLOCK_RE.search(sentence)
            or OVERRIDE_AUTH_RE.search(lowered)
            or (GENERIC_TAG_BLOCK_RE.search(sentence) and INSTR_RE.search(lowered))
        )
        sentence_b = instr_score(sentence) >= 2
        sentence_c = bool(
            binding_score(sentence, tool_names) >= 3
            or (
                surface["B"]
                and (IMPERATIVE_ANY_RE.search(sentence) or IMPERATIVE_START_RE.search(sentence))
                and _has_c_target(sentence)
            )
        )
        if not (sentence_a or sentence_b or sentence_c):
            continue
        spans.append(
            {
                "start": start,
                "end": end,
                "signals": {"A": sentence_a, "B": sentence_b, "C": sentence_c},
                "snippet": sentence[:500],
            }
        )

    structured = _structured_abc_pattern(text, tool_names)
    if structured["A"] or structured["B"] or structured["C"]:
        spans.append(
            {
                "start": 0,
                "end": len(text),
                "signals": {name: structured[name] for name in ("A", "B", "C")},
                "source": "structured_fields",
                "snippet": text[:500],
            }
        )

    for start, end, _, decoded in _iter_decoded_base64_segments(text):
        decoded_surface = _surface_abc_pattern(decoded, tool_names)
        decoded_structured = _structured_abc_pattern(decoded, tool_names)
        decoded_signals = {
            name: decoded_surface[name] or decoded_structured[name]
            for name in ("A", "B", "C")
        }
        if not any(decoded_signals.values()):
            continue
        spans.append(
            {
                "start": start,
                "end": end,
                "signals": decoded_signals,
                "source": "decoded_base64",
                "snippet": decoded[:500],
            }
        )

    return {**aggregate, "spans": spans}


def strip_injection_blocks(text: str) -> Tuple[str, int]:
    removed = 0

    def _sub(_m):
        nonlocal removed
        removed += 1
        return ""

    out = re.sub(INJECTION_BLOCK_RE, _sub, text or "")
    return out, removed


def strip_suspicious_encoded_segments(text: str, tool_names: Set[str]) -> Tuple[str, int]:
    """Remove only decodable Base64 spans whose decoded content triggers SAGE."""

    matches = list(_iter_decoded_base64_segments(text))
    if not matches:
        return text, 0
    output = text
    removed = 0
    for start, end, _, decoded in reversed(matches):
        if not detect_abc_pattern(decoded, tool_names)["trigger"]:
            continue
        output = output[:start] + "[encoded instruction removed]" + output[end:]
        removed += 1
    return output, removed

def _try_parse_structured(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Best-effort JSON/YAML parse. Only returns dict/list tool outputs; ordinary prose
    parsed by YAML as a scalar is deliberately rejected.
    """
    if not isinstance(text, str):
        return None, None
    t = text.strip()
    if not t:
        return None, None

    if t.startswith(("{", "[")):
        try:
            obj = json.loads(t)
        except Exception:
            obj = None
        if isinstance(obj, (dict, list)):
            return obj, "json"

    try:
        obj = yaml.safe_load(t)
    except yaml.YAMLError:
        return None, None
    if isinstance(obj, (dict, list)):
        return obj, "yaml"
    return None, None


def _iter_embedded_json_objects(text: str):
    """Yield non-overlapping JSON objects/arrays embedded in ordinary prose.

    AgentDojo document tools frequently return a benign text document with an
    injected JSON record appended to it.  Parsing only the complete tool output
    therefore misses the structured record and sends it through the prose
    neutralizer.  ``JSONDecoder.raw_decode`` lets us recover the record without
    making assumptions about its schema or the surrounding document format.
    """
    if not isinstance(text, str) or not text:
        return

    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(text):
        starts = [position for position in (text.find("{", cursor), text.find("[", cursor)) if position >= 0]
        if not starts:
            break
        start = min(starts)
        try:
            obj, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if not isinstance(obj, (dict, list)):
            cursor = start + 1
            continue
        end = start + consumed
        yield start, end, obj
        # Do not separately process nested objects after accepting their parent.
        cursor = end


def _iter_string_values(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_string_values(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_string_values(value)


def _neutralize_structured_values(
    obj: Any,
    tool_names: Set[str],
    source_tool: Optional[str],
    *,
    zero_out_suspicious: bool,
    force_signal_ablation: bool,
) -> Tuple[Any, int]:
    """Neutralize only string values that contribute security signals.

    A tool result may contain many records. Once one record triggers an audit,
    rewriting every benign string in the same JSON/YAML object hurts the shadow
    planner without adding protection. We aggregate A/B/C across the object to
    catch fragmented cues, then transform only signal-contributing values.
    """
    def _entry_signals(key: str, value: str) -> Dict[str, bool]:
        surface = detect_abc_pattern(value, tool_names)
        keyed = _structured_entry_signals(key, value, tool_names)
        signals = {
            name: surface[name] or keyed[name]
            for name in ("A", "B", "C")
        }
        signals["trigger"] = bool(
            (signals["A"] and signals["B"])
            or (signals["A"] and signals["C"])
            or (signals["B"] and signals["C"])
        )
        return signals

    def _scope_trigger(value: Any) -> bool:
        entries = [
            _entry_signals(key, item)
            for key, item in _iter_structured_string_entries(value)
        ]
        aggregate = {
            name: any(signals[name] for signals in entries)
            for name in ("A", "B", "C")
        }
        return bool(
            (aggregate["A"] and aggregate["B"])
            or (aggregate["A"] and aggregate["C"])
            or (aggregate["B"] and aggregate["C"])
        )

    changed_count = 0

    def _walk(value: Any, key: str = "", active_scope: bool = False) -> Any:
        nonlocal changed_count
        if isinstance(value, str):
            signals = _entry_signals(key, value)
            explicit_marker = bool(
                INJECTION_BLOCK_RE.search(value)
                or ROLE_TAG_RE.search(value)
                or OVERRIDE_AUTH_RE.search(value)
            )
            contributes_to_trigger = (active_scope or force_signal_ablation) and any(
                signals[name] for name in ("A", "B", "C")
            )
            if not (signals["trigger"] or explicit_marker or contributes_to_trigger):
                return value
            if contributes_to_trigger or (zero_out_suspicious and _suspicious_string_should_zero_out(value, tool_names)):
                output = ""
            else:
                output, _ = neutralize_text_with_meta(value, tool_names, source_tool)
            if output != value:
                changed_count += 1
            return output
        if isinstance(value, dict):
            child_scope = active_scope or _scope_trigger(value)
            return {
                child_key: _walk(item, str(child_key), child_scope)
                for child_key, item in value.items()
            }
        if isinstance(value, list):
            # A top-level list commonly represents independent benign records;
            # do not let one malicious record taint every sibling record.
            return [_walk(item, key, active_scope) for item in value]
        if isinstance(value, tuple):
            return tuple(_walk(item, key, active_scope) for item in value)
        return value

    return _walk(obj), changed_count


def neutralize_text_structured_first_with_meta(
    text: str,
    tool_names: Set[str],
    source_tool: Optional[str],
    zero_out_suspicious: bool = False,
    force_signal_ablation: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Prefer preserving structured JSON/YAML tool outputs:
      - If text parses as an object/array, neutralize its signal-bearing values.
      - If ordinary prose contains embedded JSON objects, neutralize only those
        objects while preserving the surrounding document verbatim.
      - Otherwise fall back to sentence/line neutralization.
    """
    meta: Dict[str, Any] = {"source_tool": source_tool, "structured": False}

    obj, structured_format = _try_parse_structured(text)
    if obj is not None:
        meta["structured"] = True
        meta["structured_format"] = structured_format
        try:
            obj2, changed_values = _neutralize_structured_values(
                obj,
                tool_names,
                source_tool,
                zero_out_suspicious=zero_out_suspicious,
                force_signal_ablation=force_signal_ablation,
            )
            if structured_format == "json":
                out = json.dumps(obj2, ensure_ascii=False, separators=(",", ":"))
            else:
                out = yaml.safe_dump(obj2, allow_unicode=True, sort_keys=False, default_flow_style=False).strip()
            meta["structured_values_neutralized"] = changed_values
            meta["len_chars_in"] = len(text or "")
            meta["len_chars_out"] = len(out)
            return out, meta
        except Exception as e:
            # fall back safely
            meta["structured_fallback_error"] = str(e)

    # A common injection carrier is a valid JSON record embedded in an otherwise
    # benign email, invoice, or document.  Treat each top-level embedded record
    # exactly as we treat a complete JSON tool result, then splice it back into
    # the original prose.  Benign records are left byte-for-byte unchanged.
    embedded_objects = list(_iter_embedded_json_objects(text))
    if embedded_objects:
        out = text
        changed_objects = 0
        changed_values = 0
        try:
            for start, end, embedded_obj in reversed(embedded_objects):
                embedded_obj2, object_changed_values = _neutralize_structured_values(
                    embedded_obj,
                    tool_names,
                    source_tool,
                    zero_out_suspicious=zero_out_suspicious,
                    force_signal_ablation=force_signal_ablation,
                )
                if object_changed_values <= 0:
                    continue
                replacement = json.dumps(embedded_obj2, ensure_ascii=False, separators=(",", ":"))
                out = out[:start] + replacement + out[end:]
                changed_objects += 1
                changed_values += object_changed_values
        except Exception as e:
            meta["embedded_structured_fallback_error"] = str(e)
        else:
            if changed_objects:
                meta["embedded_structured"] = True
                meta["embedded_structured_objects_neutralized"] = changed_objects
                meta["embedded_structured_values_neutralized"] = changed_values
                meta["len_chars_in"] = len(text or "")

                # Structured ablation is intentionally precise.  Only invoke the
                # prose fallback when suspicious A/B/C cues remain outside the
                # recovered JSON object (for example, in a wrapper instruction).
                if detect_abc_pattern(out, tool_names)["trigger"]:
                    out, fallback_meta = neutralize_text_with_meta(out, tool_names, source_tool)
                    meta["embedded_residual_prose_fallback"] = True
                    meta["embedded_residual_prose_meta"] = fallback_meta
                else:
                    meta["embedded_residual_prose_fallback"] = False
                meta["len_chars_out"] = len(out)
                return out, meta

    out, meta2 = neutralize_text_with_meta(text, tool_names, source_tool)
    meta.update(meta2)
    return out, meta



def neutralize_text_with_meta(
    text: str,
    tool_names: Set[str],
    source_tool: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Generic neutralize:
      0) strip obvious wrapper blocks (optional)
      1) sentence-level filtering: drop instruction/binding sentences, keep factual ones
      2) line-level cleanup: drop remaining binding / imperative-only lines
      3) high-risk L2 fallback: keep only structured-ish lines when very unstructured
    """
    # print(f"tool_names={tool_names}")
    meta: Dict[str, Any] = {"source_tool": source_tool}
    if not text:
        return "", {"empty": True, **meta}

    raw = text
    decoded_stripped, removed_encoded_segments = strip_suspicious_encoded_segments(raw, tool_names)
    t1, removed_blocks = strip_injection_blocks(decoded_stripped)
    meta["removed_encoded_segments"] = removed_encoded_segments
    meta["removed_injection_blocks"] = removed_blocks
    meta["len_chars_in"] = len(raw)
    meta["len_chars_after_block_strip"] = len(t1)

    kept_sent: List[str] = []
    dropped_sent_instr = 0
    dropped_sent_bind = 0
    stripped_role_tails = 0

    input_sentences = split_sentences(t1)
    for sent in input_sentences:
        original_sent = sent
        sent = strip_role_marker_tail(sent)
        if sent != original_sent:
            stripped_role_tails += 1
        if not sent:
            dropped_sent_instr += 1
            continue

        b = binding_score(sent, tool_names)
        i = instr_score(sent)
        f = fact_score(sent)


        if b >= 3:
            dropped_sent_bind += 1
            continue
        direct_action_with_target = bool(
            i >= 3
            and _has_c_target(sent)
            and (IMPERATIVE_START_RE.search(sent) or DEONTIC_RE.search(sent) or PROCEDURE_RE.search(sent))
        )
        if direct_action_with_target:
            dropped_sent_instr += 1
            continue
        if i >= 3 and f == 0:
            dropped_sent_instr += 1
            continue
        if i >= 3 and URL_RE.search(sent) and f <= 1:
            dropped_sent_instr += 1
            continue
        kept_sent.append(sent)

    t2 = "\n".join(kept_sent).strip()
    meta["sentence_in"] = len(input_sentences)
    meta["sentence_out"] = len(kept_sent)
    meta["dropped_sentence_binding"] = dropped_sent_bind
    meta["dropped_sentence_instruction"] = dropped_sent_instr
    meta["stripped_role_tails"] = stripped_role_tails

    lines = split_lines(t2)
    kept_lines: List[str] = []
    dropped_line_bind = 0
    dropped_line_imp = 0

    for ln in lines:
        ln = strip_role_marker_tail(ln)
        if not ln:
            dropped_line_imp += 1
            continue
        b = binding_score(ln, tool_names)
        i = instr_score(ln)
        f = fact_score(ln)

        if b >= 3:
            dropped_line_bind += 1
            continue
        direct_action_with_target = bool(
            i >= 3
            and _has_c_target(ln)
            and (IMPERATIVE_START_RE.search(ln) or DEONTIC_RE.search(ln) or PROCEDURE_RE.search(ln))
        )
        if direct_action_with_target:
            dropped_line_imp += 1
            continue
        if i >= 3 and f == 0 and not is_structuredish(ln):
            dropped_line_imp += 1
            continue

        kept_lines.append(ln)

    t3 = "\n".join(kept_lines).strip()
    meta["lines_in"] = len(lines)
    meta["lines_out"] = len(kept_lines)
    meta["dropped_line_binding"] = dropped_line_bind
    meta["dropped_line_imperative"] = dropped_line_imp

     # ---------------------------
    # Content-driven L2 fallback (NO high_risk)
    # ---------------------------
    structured_lines = sum(1 for ln in kept_lines if is_structuredish(ln))
    field_ratio = structured_lines / max(1, len(kept_lines))

    # length / structure heuristics
    long_text = len(t3) > 2000 or len(kept_lines) > 60
    very_unstructured = (len(t3) > 800 and field_ratio < 0.25)

    sentence_drop_ratio = (dropped_sent_bind + dropped_sent_instr) / max(1, len(input_sentences))
    line_drop_ratio = (dropped_line_bind + dropped_line_imp) / max(1, len(lines))
    drop_ratio = max(sentence_drop_ratio, line_drop_ratio)

    residual_suspicious = any(
        binding_score(ln, tool_names) >= 3
        or bool(OVERRIDE_AUTH_RE.search(ln))
        or bool(ROLE_TAG_RE.search(ln))
        for ln in kept_lines
    )

    do_l2ish = bool(
        residual_suspicious
        and ((long_text and very_unstructured) or (drop_ratio >= 0.6 and len(t3) > 500))
    )

    meta["field_ratio"] = round(field_ratio, 3)
    meta["drop_ratio"] = round(drop_ratio, 3)
    meta["residual_suspicious"] = residual_suspicious
    meta["do_l2ish"] = do_l2ish

    if do_l2ish:
        kept2 = [ln for ln in kept_lines if is_structuredish(ln)]
        meta["l2_dropped_free_text_lines"] = len(kept_lines) - len(kept2)
        out = "\n".join(kept2).strip()
    else:
        out = t3

    # The filtering pipeline normalizes sentence whitespace. If nothing was
    # removed, return the exact original value so benign tool data is not
    # rewritten merely because it passed through the defense.
    if not (
        removed_blocks
        or removed_encoded_segments
        or dropped_sent_bind
        or dropped_sent_instr
        or stripped_role_tails
        or dropped_line_bind
        or dropped_line_imp
        or do_l2ish
    ):
        out = raw

    meta["len_chars_out"] = len(out)
    meta["num_lines_out"] = len(split_lines(out))
    if not out:
        meta["empty_after_neutralize"] = True
    return out, meta

def neutralize_text(text: str, tool_names: Set[str], source_tool: Optional[str]) -> str:
    out, _ = neutralize_text_with_meta(text, tool_names, source_tool)
    return out

def neutralize_deep(
    obj: Any,
    tool_names: List[str],
    source_tool: Optional[str],
    *,
    zero_out_suspicious: bool = False,
) -> Any:
    if isinstance(obj, str):
        if zero_out_suspicious and _suspicious_string_should_zero_out(obj, set(tool_names)):
            return ""
        out, _meta = neutralize_text_with_meta(obj, set(tool_names), source_tool)
        return out
    if isinstance(obj, dict):
        return {k: neutralize_deep(v, tool_names, source_tool, zero_out_suspicious=zero_out_suspicious) for k, v in obj.items()}
    if isinstance(obj, list):
        return [neutralize_deep(v, tool_names, source_tool, zero_out_suspicious=zero_out_suspicious) for v in obj]
    if isinstance(obj, tuple):
        return tuple(neutralize_deep(v, tool_names, source_tool, zero_out_suspicious=zero_out_suspicious) for v in obj)
    return obj



def neutralize_tool_message_error_and_args_inplace(
    msg: ChatMessage,
    tool_names: List[str],
) -> bool:
    """
    Neutralize injection that may appear in:
      - tool message "error" (string)
      - tool message tool_call.args (ANY nested string fields)
    Returns True if anything changed.
    """
    if msg.get("role") != "tool":
        return False

    changed = False

    tc_obj = msg.get("tool_call")
    if isinstance(tc_obj, dict):
        src_tool = tc_obj.get("function") or tc_obj.get("name")
        tc_args = tc_obj.get("args") or {}
    else:
        src_tool = getattr(tc_obj, "function", None) or getattr(tc_obj, "name", None)
        tc_args = getattr(tc_obj, "args", None) or {}
    src_tool = src_tool or msg.get("name") or "tool"

    err = msg.get("error")
    if isinstance(err, str) and err.strip():
        err2, _meta = neutralize_text_with_meta(err, set(tool_names), src_tool)
        if err2 != err:
            msg["error"] = err2
            changed = True

    if tc_args is not None:
        tc_args2 = neutralize_deep(tc_args, tool_names, src_tool)
        if tc_args2 != tc_args:
            tc_args = tc_args2
            changed = True

        if isinstance(tc_obj, dict):
            tc_obj["args"] = tc_args
            msg["tool_call"] = tc_obj
        else:
            try:
                setattr(tc_obj, "args", tc_args)
            except Exception:
                pass

    return changed

def args_equivalent_under_neutralization(
    a: Any,
    b: Any,
    tool_names: Set[str],
    source_tool: Optional[str],
) -> bool:
    """
    Two argument objects are considered equivalent if they become identical after applying
    the SAME neutralization transform. This handles "identifier contains injection text"
    cases (channel/user/file names etc.) without special-casing any field.
    """
    try:
        a_norm = neutralize_deep(a, list(tool_names), source_tool)
        b_norm = neutralize_deep(b, list(tool_names), source_tool)
        return _canonicalize_for_compare(a_norm) == _canonicalize_for_compare(b_norm)

    except Exception:
        return False

__all__ = [
    "neutralize_text",
    "neutralize_text_with_meta",
    "neutralize_deep",
    "neutralize_tool_message_error_and_args_inplace",
    "args_equivalent_under_neutralization",
    "detect_abc_evidence",
    "detect_abc_pattern",
    "extract_fragment_correlation_ids",
    "strip_correlated_fragment_blocks",
    "neutralize_text_structured_first_with_meta",
]
