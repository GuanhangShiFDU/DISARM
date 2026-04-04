from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .disarm_patterns import (
    BASE64_RE,
    INJECTION_BLOCK_RE,
    MONEY_RE,
    ROLE_TAG_RE,
    URL_RE,
    _canonicalize_for_compare,
    binding_score,
    detect_abc_pattern,
    fact_score,
    instr_score,
    is_structuredish,
    split_lines,
    split_sentences,
    strip_role_marker_tail,
)


def _strip_base64(text: str) -> Tuple[str, int]:
    count = 0

    def _sub(_m):
        nonlocal count
        count += 1
        return "[ENCODED_REMOVED]"

    return BASE64_RE.sub(_sub, text), count


def strip_injection_blocks(text: str) -> Tuple[str, int]:
    removed = 0

    def _sub(_m):
        nonlocal removed
        removed += 1
        return ""

    out = re.sub(INJECTION_BLOCK_RE, _sub, text or "")
    return out, removed


def _try_parse_json(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t or not (t.startswith("{") or t.startswith("[")):
        return None
    try:
        obj = json.loads(t)
    except Exception:
        return None
    return obj if isinstance(obj, (dict, list)) else None


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
    if instr_score(s) > 0 and (URL_RE.search(s) or MONEY_RE.search(s)):
        return True

    return False


def neutralize_text_with_meta(
    text: str,
    tool_names: Set[str],
    source_tool: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"source_tool": source_tool}
    if not text:
        return "", {"empty": True, **meta}

    raw = text
    t0, removed_encoded = _strip_base64(raw)
    meta["removed_encoded_segments"] = removed_encoded

    t1, removed_blocks = strip_injection_blocks(t0)
    meta["removed_injection_blocks"] = removed_blocks
    meta["len_chars_in"] = len(raw)
    meta["len_chars_after_block_strip"] = len(t1)

    kept_sent: List[str] = []
    dropped_sent_instr = 0
    dropped_sent_bind = 0

    for sent in split_sentences(t1):
        sent = strip_role_marker_tail(sent)
        if not sent:
            dropped_sent_instr += 1
            continue

        b = binding_score(sent, tool_names)
        i = instr_score(sent)
        f = fact_score(sent)

        if b >= 3:
            dropped_sent_bind += 1
            continue
        if i >= 3 and f == 0:
            dropped_sent_instr += 1
            continue
        if i >= 3 and URL_RE.search(sent) and f <= 1:
            dropped_sent_instr += 1
            continue

        kept_sent.append(sent)

    t2 = "\n".join(kept_sent).strip()
    meta["sentence_in"] = len(split_sentences(t1))
    meta["sentence_out"] = len(kept_sent)
    meta["dropped_sentence_binding"] = dropped_sent_bind
    meta["dropped_sentence_instruction"] = dropped_sent_instr

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
        if i >= 3 and f == 0 and not is_structuredish(ln):
            dropped_line_imp += 1
            continue

        kept_lines.append(ln)

    t3 = "\n".join(kept_lines).strip()
    meta["lines_in"] = len(lines)
    meta["lines_out"] = len(kept_lines)
    meta["dropped_line_binding"] = dropped_line_bind
    meta["dropped_line_imperative"] = dropped_line_imp

    structured_lines = sum(1 for ln in kept_lines if is_structuredish(ln))
    field_ratio = structured_lines / max(1, len(kept_lines))
    long_text = len(t3) > 2000 or len(kept_lines) > 60
    very_unstructured = (len(t3) > 800 and field_ratio < 0.25)

    dropped_total = dropped_sent_bind + dropped_sent_instr + dropped_line_bind + dropped_line_imp
    kept_total = max(1, len(kept_sent) + len(kept_lines))
    drop_ratio = dropped_total / kept_total

    do_l2ish = bool((long_text and very_unstructured) or (drop_ratio >= 0.6 and len(t3) > 500))
    meta["field_ratio"] = round(field_ratio, 3)
    meta["drop_ratio"] = round(drop_ratio, 3)
    meta["do_l2ish"] = do_l2ish

    if do_l2ish:
        kept2 = [ln for ln in kept_lines if is_structuredish(ln)]
        meta["l2_dropped_free_text_lines"] = len(kept_lines) - len(kept2)
        out = "\n".join(kept2).strip()
    else:
        out = t3

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


def neutralize_text_structured_first_with_meta(
    text: str,
    tool_names: Set[str],
    source_tool: Optional[str],
    zero_out_suspicious: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"source_tool": source_tool, "structured": False}

    obj = _try_parse_json(text)
    if obj is not None:
        meta["structured"] = True
        try:
            obj2 = neutralize_deep(obj, list(tool_names), source_tool, zero_out_suspicious=zero_out_suspicious)
            out = json.dumps(obj2, ensure_ascii=False, separators=(",", ":"))
            meta["len_chars_in"] = len(text or "")
            meta["len_chars_out"] = len(out)
            return out, meta
        except Exception as e:
            meta["structured_fallback_error"] = str(e)

    out, meta2 = neutralize_text_with_meta(text, tool_names, source_tool)
    meta.update(meta2)
    return out, meta


def neutralize_tool_message_error_and_args_inplace(
    msg: "ChatMessage",
    tool_names: List[str],
) -> bool:
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
    "neutralize_text_structured_first_with_meta",
]