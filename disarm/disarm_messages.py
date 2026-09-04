from __future__ import annotations

import copy
import re
import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    text_content_block_from_string,
    get_text_content_as_str,
)
from .disarm_neutralize import neutralize_text


_NOISE_RE = re.compile(r"[\s\.\,\;\:\'\"\`\u2018\u2019\u201c\u201d]+")
_YAML_WRAP_FIX_RE = re.compile(r"\n[ \t]+")
_YAML_DBL_QUOTE_RE = re.compile(r"''")        # YAML single-quote escape
_PARA_SENTINEL = "\u000b"
_YAML_PRETTY_INDENT_RE = re.compile(r"\n[ \t]{2,}\S")
_CODE_FENCE_RE = re.compile(r"```")
_FREE_TEXT_KEYS = {
    "body", "text", "content", "message", "prompt", "query", "description", "summary"
}

def _looks_like_yaml_pretty(s: str) -> bool:
    """
    Heuristic: detect YAML pretty/wrapped scalar produced by tools (indent, doubled single quotes, hard wraps).
    We only want to fix cases like:
      - lines indented after newline
      - YAML single-quote escaping: '' -> '
      - lots of wrapped lines (many '\n' without meaning)
    """
    if not isinstance(s, str) or "\n" not in s:
        return False
    if _CODE_FENCE_RE.search(s):
        return False  # code fences: newline is meaningful

    # Strong signals
    has_indent = _YAML_PRETTY_INDENT_RE.search(s) is not None
    has_yaml_quote_escape = "''" in s  # YAML single-quoted escaping

    # More structure-based checks (avoid harming real paragraphs)
    lines = s.splitlines()
    if len(lines) < 2:
        return False

    # Ratio of lines starting with indentation (YAML pretty often does this)
    indented_lines = sum(1 for ln in lines[1:] if ln.startswith(("  ", "\t")))
    indent_ratio = indented_lines / max(1, (len(lines) - 1))

    # If it has strong YAML signals, accept
    if has_indent or has_yaml_quote_escape:
        return True

    # Otherwise require "pretty-wrap like": many short-ish lines, mostly not paragraph breaks
    # (this is weaker; keep conservative)
    avg_len = sum(len(ln) for ln in lines) / len(lines)
    has_blank_line = any(ln.strip() == "" for ln in lines)
    if indent_ratio >= 0.6 and 20 <= avg_len <= 120 and not has_blank_line:
        return True

    return False


def _maybe_fix_yaml_pretty(s: str, key: Optional[str] = None) -> str:
    """
    Apply pretty fix only when it *looks like* YAML pretty.
    For free-text keys, be extra conservative.
    """
    if not s:
        return s

    # 对自由文本字段更保守：必须有“强信号”才修
    if key in _FREE_TEXT_KEYS:
        if "\n" not in s:
            return s
        if not (_YAML_PRETTY_INDENT_RE.search(s) or "''" in s):
            return s

    if not _looks_like_yaml_pretty(s):
        return s

    return _fix_yaml_pretty(s)

def _fix_yaml_pretty(s: str) -> str:
    if not s:
        return s

    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # 1) remove YAML indentation after newline: "\n  You" -> "\nYou"
    s = _YAML_WRAP_FIX_RE.sub("\n", s)

    # 2) YAML single-quoted string escaping: '' -> '
    s = _YAML_DBL_QUOTE_RE.sub("'", s)

    # 3) protect REAL paragraph breaks (blank line): "\n\n" (or "\n \n") => sentinel
    s = re.sub(r"\n[ \t]*\n+", _PARA_SENTINEL, s)

    # 4) remaining single '\n' is soft-wrap => convert to single space
    s = s.replace("\n", " ")

    # 5) restore paragraph breaks as ONE '\n'
    s = s.replace(_PARA_SENTINEL, "\n")

    # 6) collapse extra spaces
    s = re.sub(r"[ \t]{2,}", " ", s)

    return s

def _skeleton(s: str) -> str:
    """
    Aggressive-ish identifier skeleton:
    remove whitespace + common punctuation noise (including '.' in the middle).
    Lowercase for stability.
    """
    if not s:
        return ""
    return _NOISE_RE.sub("", s).strip().lower()

def _is_identifier_like(s: str) -> bool:
    """
    Avoid rewriting large free-text bodies.
    Heuristic: short, low whitespace, looks like a name/id/path.
    """
    if not isinstance(s, str):
        return False
    t = s.strip()
    if not t or len(t) > 200:
        return False
    # too many spaces => likely natural language
    if t.count(" ") >= 6:
        return False
    return True

def resolve_args_from_history(
    messages: Sequence[ChatMessage],
    args: Dict[str, Any],
    tool_names: Set[str],
    source_tool: Optional[str],
) -> Dict[str, Any]:
    # print(f"messages={messages}")
    cands = _extract_candidate_strings_from_history(messages)
    repaired = _resolve_strings_via_candidates(args, cands, tool_names, source_tool)
    return repaired if isinstance(repaired, dict) else dict(args)

def tool_name(tc: Any) -> Optional[str]:
    if hasattr(tc, "function"):
        return getattr(tc, "function")
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, str):
            return fn
        if isinstance(fn, dict) and "name" in fn:
            return fn["name"]
        if "name" in tc and isinstance(tc["name"], str):
            return tc["name"]
    return None


def tool_args(tc: Any) -> Dict[str, Any]:
    if hasattr(tc, "args"):
        a = getattr(tc, "args")
        return dict(a) if isinstance(a, dict) else {}
    if isinstance(tc, dict):
        a = tc.get("args") or tc.get("arguments") or {}
        if isinstance(a, str):
            try:
                return json.loads(a)
            except Exception:
                return {}
        return dict(a) if isinstance(a, dict) else {}
    return {}


def first_tool_name(tool_calls: List[Any]) -> Optional[str]:
    if not tool_calls:
        return None
    return tool_name(tool_calls[0])


def tool_call_to_runtime_format(tc: Any) -> Tuple[str, Dict[str, Any]]:
    return (tool_name(tc) or ""), tool_args(tc)


def runtime_tool_names(runtime: FunctionsRuntime) -> Set[str]:
    if hasattr(runtime, "tools"):
        return {t.name for t in runtime.tools}
    if hasattr(runtime, "functions"):
        return set(runtime.functions.keys())
    return set()


def get_last_assistant_tool_calls(messages: Sequence[ChatMessage]) -> List[Any]:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            tcs = msg.get("tool_calls") or []
            return list(tcs)
    return []


def mk_system(text: str) -> ChatSystemMessage:
    return ChatSystemMessage(role="system", content=[text_content_block_from_string(text)])


def mk_tool_result(tool_call: FunctionCall, returned_text: str) -> ChatToolResultMessage:
    return ChatToolResultMessage(
        role="tool",
        tool_call_id=str(tool_call.id),
        tool_call=tool_call,
        content=[text_content_block_from_string(returned_text)],
        error=None,
    )


def prune_last_assistant_tool_calls_inplace(
    messages: List[ChatMessage],
    keep_tool: Optional[str],
) -> bool:
    """
    Keep ONLY tool_calls whose tool name == keep_tool in the LAST assistant message that has tool_calls.
    This prevents executing sibling tool_calls that were co-produced in the same assistant message.
    """
    if not keep_tool:
        return False

    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        if not tcs:
            continue

        kept: List[Any] = []
        for tc in list(tcs):
            tn = tool_name(tc) or tool_call_to_runtime_format(tc)[0]
            if tn == keep_tool:
                kept.append(tc)

        if kept:
            m["tool_calls"] = kept
            return True
        return False

    return False


# def remove_last_blacklist_attempt(messages: List[ChatMessage], blacklist: Set[str]) -> List[ChatMessage]:
#     """
#     删除“最近一次 assistant 发出的 blacklist tool_call”以及它紧随其后的 tool 回包（若存在）。
#     防止 dangling tool_call_id / 防止危险调用污染后续。
#     """
#     msgs = list(messages)

#     i = len(msgs) - 1
#     while i >= 0:
#         m = msgs[i]
#         if m.get("role") == "assistant" and m.get("tool_calls"):
#             tcs = m.get("tool_calls") or []
#             first = tcs[0] if tcs else None
#             fn = getattr(first, "function", None)
#             if fn in blacklist:
#                 ids = [getattr(tc, "id", None) for tc in tcs]
#                 ids = {str(x) for x in ids if x is not None}

#                 msgs.pop(i)

#                 j = i
#                 while j < len(msgs):
#                     mm = msgs[j]
#                     if mm.get("role") == "tool" and str(mm.get("tool_call_id")) in ids:
#                         msgs.pop(j)
#                         continue
#                     j += 1
#                 break
#         i -= 1

#     return msgs

def remove_last_unwhitelisted_attempt(messages: List[ChatMessage], whitelist: Set[str]) -> List[ChatMessage]:
    """
    Remove only the unwhitelisted tool_calls from the last assistant tool_calls message,
    and remove ONLY the tool results that correspond to those removed tool_call_ids.
    Keep whitelisted sibling tool_calls and their tool results.
    """
    msgs = list(messages)

    # find last assistant with tool_calls
    i = len(msgs) - 1
    while i >= 0:
        m = msgs[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = list(m.get("tool_calls") or [])

            # split tool_calls into keep vs drop
            keep_tcs = []
            drop_ids: Set[str] = set()

            for tc in tcs:
                fn = getattr(tc, "function", None) if hasattr(tc, "function") else None
                if not fn and isinstance(tc, dict):
                    fn = tc.get("function") or (tc.get("function", {}) or {}).get("name")

                tcid = getattr(tc, "id", None) if hasattr(tc, "id") else (tc.get("id") if isinstance(tc, dict) else None)
                if fn and fn not in whitelist:
                    if tcid is not None:
                        drop_ids.add(str(tcid))
                else:
                    keep_tcs.append(tc)

            # if nothing unwhitelisted, do nothing
            if not drop_ids:
                return msgs

            # mutate assistant tool_calls: keep only non-unwhitelisted
            if keep_tcs:
                msgs[i]["tool_calls"] = keep_tcs
            else:
                # if all tool_calls were unwhitelisted, remove the whole assistant message
                msgs.pop(i)

            # remove only tool results for dropped ids
            j = i
            while j < len(msgs):
                mm = msgs[j]
                if mm.get("role") == "tool" and str(mm.get("tool_call_id")) in drop_ids:
                    msgs.pop(j)
                    continue
                j += 1

            return msgs

        i -= 1

    return msgs



def fill_dangling_tool_results_inplace(messages: List[ChatMessage], neutral_text: str = "TOOL_RESULT_OK") -> None:
    """
    OpenAI tools 协议：assistant(tool_calls=[...]) 后必须跟对应 tool(tool_call_id=...)，否则会 400。
    这里只补齐“最后一个有 tool_calls 的 assistant”的缺失 tool 结果。
    """
    last_idx = None
    last_tcs: List[Any] = []
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            last_idx = i
            last_tcs = list(m.get("tool_calls") or [])
            break

    if last_idx is None or not last_tcs:
        return

    responded: Set[str] = set()
    for m in messages[last_idx + 1:]:
        if m.get("role") == "tool" and m.get("tool_call_id") is not None:
            responded.add(str(m["tool_call_id"]))

    inserts: List[ChatToolResultMessage] = []
    for tc in last_tcs:
        if not isinstance(tc, FunctionCall):
            continue
        tcid = str(tc.id)
        if tcid in responded:
            continue
        inserts.append(mk_tool_result(tc, neutral_text))

    if not inserts:
        return

    insert_pos = last_idx + 1
    messages[insert_pos:insert_pos] = inserts


def strip_last_assistant_tool_call_inplace(messages: List[ChatMessage], expected_tool: str) -> bool:
    """
    counterfactual/shadow：移除最后一条 assistant(tool_calls=expected_tool) 以便重新决策。
    """
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = m.get("tool_calls") or []
            fn = tool_name(tcs[0]) if tcs else None
            if fn == expected_tool:
                messages.pop(i)
                return True
            return False
    return False


def _set_tool_args_inplace(tc: Any, new_args: Dict[str, Any]) -> bool:
    """
    Mutate tool call args in-place.
    Supports:
      - agentdojo.functions_runtime.FunctionCall (has .args)
      - dict style tool call (args/arguments)
    """
    if tc is None:
        return False

    if hasattr(tc, "args"):
        setattr(tc, "args", dict(new_args))
        return True

    if isinstance(tc, dict):
        if "args" in tc and isinstance(tc["args"], dict):
            tc["args"] = dict(new_args)
            return True
        if "arguments" in tc:
            if isinstance(tc["arguments"], str):
                tc["arguments"] = json.dumps(new_args, ensure_ascii=False)
                return True
            tc["arguments"] = dict(new_args)
            return True
        tc["args"] = dict(new_args)
        return True

    return False


def replace_last_tool_call_args_inplace(
    messages: List[ChatMessage],
    expected_tool: str,
    new_args: Dict[str, Any],
) -> bool:
    """
    Find the last assistant message with tool_calls; if its first tool matches expected_tool,
    replace its first tool call's args with new_args. In-place.
    """
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        if not tcs:
            continue

        tc0 = tcs[0]
        tool = tool_name(tc0)

        if tool != expected_tool:
            return False

        return _set_tool_args_inplace(tc0, new_args)

    return False


def replace_last_tool_call_tool_and_args_inplace(
    messages: List[ChatMessage],
    new_tool: str,
    new_args: Dict[str, Any],
) -> bool:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        if not tcs:
            continue

        tc0 = tcs[0]

        if hasattr(tc0, "function"):
            setattr(tc0, "function", new_tool)
        elif isinstance(tc0, dict):
            fn = tc0.get("function")
            if isinstance(fn, dict) and "name" in fn:
                fn["name"] = new_tool
            else:
                tc0["function"] = new_tool

        return _set_tool_args_inplace(tc0, new_args)

    return False


def replace_last_assistant_tool_calls_inplace(
    messages: List[ChatMessage],
    new_tool_calls: Sequence[Any],
) -> bool:
    """Replace the complete most-recent assistant action batch."""

    if not new_tool_calls:
        return False
    for message in reversed(messages):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        message["tool_calls"] = copy.deepcopy(list(new_tool_calls))
        return True
    return False


def strip_last_assistant_tool_call_any_inplace(messages: List[ChatMessage], expected_tool: str) -> bool:
    """
    shadow：移除最后一条 assistant，只要它的 tool_calls 里【任意一个】匹配 expected_tool 就移除。
    用于处理注入塞进 tool_calls[1]/[2] 的情况，避免 400。
    """
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = list(m.get("tool_calls") or [])
            for tc in tcs:
                if tool_name(tc) == expected_tool:
                    messages.pop(i)
                    return True
            return False
    return False

_BULLET_SPAN_RE = re.compile(r"(?ms)^\- (.*?)(?=\n\- |\Z)")

def _unquote_once(s: str) -> str:
    # IMPORTANT: do NOT strip; keep raw spaces exactly
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {"'", '"'}:
        return s[1:-1]
    return s

def _extract_candidate_strings_from_history(messages: Sequence[ChatMessage]) -> List[Tuple[str, str]]:
    """
    Extract *raw* candidates from tool outputs + user messages.
    Returns: List[(raw_text, role)] where role in {"tool","user"}.
    """
    out: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    def _push(raw: str, role: str) -> None:
        if raw is None or raw == "":
            return
        key = (raw, role)
        if key in seen:
            return
        seen.add(key)
        out.append(key)

    def _extract_from_text(txt: str, role: str) -> None:
        if not txt:
            return

        # 1) full bullet spans first (raw, byte-preserving)
        for m in _BULLET_SPAN_RE.finditer(txt):
            item_raw = m.group(1)
            if item_raw:
                _push(item_raw, role)
                uq = _unquote_once(item_raw)
                if uq != item_raw:
                    _push(uq, role)

        # 2) secondary: raw non-empty lines (NO strip)
        for ln in txt.splitlines():
            if ln.strip() == "":
                continue
            _push(ln, role)

    for m in messages:
        role = m.get("role")
        if role not in {"tool", "user"}:
            continue

        txt = get_text_content_as_str(m.get("content") or []) or ""
        _extract_from_text(txt, role)

        if role == "tool":
            err = m.get("error")
            if isinstance(err, str) and err:
                _extract_from_text(err, role)

    return out



def _resolve_strings_via_candidates(
    obj: Any,
    candidates: List[Tuple[str, str]],
    tool_names: Set[str],
    source_tool: Optional[str],
) -> Any:
    """
    Key idea:
      - Build match keys from a trimmed view (stable)
      - Store *raw candidate* as value (byte-perfect)
    """
    norm_to_best: Dict[str, str] = {}
    skel_to_best: Dict[str, str] = {}

    def _skeleton_key(s: str) -> str:
        # you can keep your existing skeleton; key only
        # IMPORTANT: this should be derived from a trimmed view, not raw
        return _skeleton(s)

    for raw_c, role in candidates:
        # key uses trimmed view, but value stores raw
        c_trim = raw_c.strip()
        if not c_trim:
            continue

        cn = neutralize_text(c_trim, tool_names, source_tool) or ""
        if cn and cn != "TOOL_RESULT_OK" and cn not in norm_to_best:
            norm_to_best[cn] = raw_c  # ✅ store raw

        cs = _skeleton_key(c_trim)
        if cs and cs not in skel_to_best:
            skel_to_best[cs] = raw_c  # ✅ store raw

    def rec(x: Any, key: Optional[str] = None) -> Any:
        if isinstance(x, str):
            raw = x
            v_match = raw.strip()

            if len(v_match) > 20000:
                return x

            # 1) norm match
            vn = neutralize_text(v_match, tool_names, source_tool) or ""
            best = norm_to_best.get(vn)
            if best is not None:
                return best
            # 2) skeleton match
            vs = _skeleton(v_match)
            best2 = skel_to_best.get(vs)
            if best2 is not None:
                return best2

            if key in _FREE_TEXT_KEYS and v_match:
                v_low = v_match.lower()
                for raw_c, role in candidates:
                    if role != "user":
                        continue
                    raw_low = raw_c.lower()
                    idx = raw_low.find(v_low)
                    if idx != -1:
                        # return exact slice from user raw text
                        return raw_c[idx : idx + len(v_match)]

            return x

        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                vv = rec(v, key=str(k))
                if isinstance(vv, str):
                    vv = _maybe_fix_yaml_pretty(vv, key=str(k))
                out[k] = vv
            return out
        if isinstance(x, list):
            return [rec(v) for v in x]
        if isinstance(x, tuple):
            return tuple(rec(v) for v in x)
        return x

    return rec(obj)

__all__ = [
    "tool_name",
    "tool_args",
    "first_tool_name",
    "tool_call_to_runtime_format",
    "runtime_tool_names",
    "get_last_assistant_tool_calls",
    "mk_system",
    "mk_tool_result",
    "prune_last_assistant_tool_calls_inplace",
    "remove_last_unwhitelisted_attempt",
    "fill_dangling_tool_results_inplace",
    "strip_last_assistant_tool_call_inplace",
    "replace_last_tool_call_args_inplace",
    "replace_last_tool_call_tool_and_args_inplace",
    "replace_last_assistant_tool_calls_inplace",
    "strip_last_assistant_tool_call_any_inplace",
    "resolve_args_from_history",
]
