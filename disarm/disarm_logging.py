from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set

from agentdojo.types import ChatMessage, get_text_content_as_str


def _short(s: str, n: int = 180) -> str:
    s = (s or "").replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"...(len={len(s)})"


def _pretty_args(d: Dict[str, Any], n: int = 240) -> str:
    try:
        import json
        s = json.dumps(d, ensure_ascii=False)
    except Exception:
        s = str(d)
    return _short(s, n)


def disarm_trace(tag: str, lines: List[str], enabled: bool = True) -> None:
    if not enabled:
        return
    print("\n" + "=" * 100)
    print(f"[DISARM] {tag}")
    for ln in lines:
        print(ln)
    print("=" * 100 + "\n")


def _msg_brief(m: ChatMessage) -> str:
    role = m.get("role")

    if role == "assistant":
        tcs = m.get("tool_calls") or []
        if tcs:
            fn = getattr(tcs[0], "function", None)
            return f"assistant(tool_calls={len(tcs)} first={fn})"
        txt = get_text_content_as_str(m.get("content", [])) or ""
        return f"assistant(text={_short(txt)})"

    if role == "tool":
        tcid = m.get("tool_call_id")
        tc = m.get("tool_call")
        fn = getattr(tc, "function", None) if tc is not None else None
        txt = get_text_content_as_str(m.get("content", [])) or ""
        return f"tool(id={tcid} fn={fn} text={_short(txt)})"

    if role == "system":
        txt = get_text_content_as_str(m.get("content", [])) or ""
        return f"system(text={_short(txt)})"

    if role == "user":
        txt = get_text_content_as_str(m.get("content", [])) or ""
        return f"user(text={_short(txt)})"

    return str(role)


def disarm_trace_messages(tag: str, messages: Sequence[ChatMessage], enabled: bool = True, tail: int = 8) -> None:
    if not enabled:
        return
    start = max(0, len(messages) - tail)
    lines = [f"messages_total={len(messages)} showing_last={tail}"]
    for i in range(start, len(messages)):
        lines.append(f"  [{i}] {_msg_brief(messages[i])}")
    disarm_trace(tag, lines, enabled=enabled)


def disarm_trace_integrity(tag: str, messages: Sequence[ChatMessage], enabled: bool = True) -> None:
    if not enabled:
        return

    assistant_tool_calls: Dict[str, int] = {}
    tool_responses: Set[str] = set()

    for idx, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tcid = getattr(tc, "id", None)
                assistant_tool_calls[str(tcid)] = idx
        if m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            if tcid is not None:
                tool_responses.add(str(tcid))

    dangling = set(assistant_tool_calls.keys()) - tool_responses
    disarm_trace(tag, [
        f"assistant_tool_calls={len(assistant_tool_calls)} tool_responses={len(tool_responses)}",
        f"dangling_tool_call_ids={sorted(list(dangling))[:20]}{'...' if len(dangling) > 20 else ''}",
    ], enabled=enabled)


__all__ = ["_pretty_args", "_short", "disarm_trace", "disarm_trace_messages", "disarm_trace_integrity"]
