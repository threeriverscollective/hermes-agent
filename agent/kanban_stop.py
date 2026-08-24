"""Turn-end guard for kanban workers.

Kanban workers must end with ``kanban_complete`` or ``kanban_block``. Models
(especially GLM / Qwen families) sometimes narrate the next step
("Let me write the report now") and stop with ``finish_reason=stop`` and no
tool calls. Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``.

This module is policy-only: when a kanban worker tries to finish without a
terminal board tool, return a bounded synthetic nudge so the conversation
loop continues instead of exiting.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional


_TERMINAL_KANBAN_TOOLS = frozenset({"kanban_complete", "kanban_block"})

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_stop_nudge_enabled() -> bool:
    """Return whether the kanban stop-guard is active for this process.

    On when ``HERMES_KANBAN_TASK`` is set (dispatcher-spawned worker), unless
    ``HERMES_KANBAN_STOP_NUDGE`` explicitly disables it.
    """
    env = os.environ.get("HERMES_KANBAN_STOP_NUDGE")
    if env is not None and env.strip().lower() in {"0", "false", "no", "off"}:
        return False
    task = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    return bool(task)


def _tool_call_name(tc: Any) -> str:
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return str(fn.get("name") or "")
        return str(tc.get("name") or "")
    fn = getattr(tc, "function", None)
    if fn is not None:
        return str(getattr(fn, "name", "") or "")
    return str(getattr(tc, "name", "") or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already invoked a terminal kanban tool."""
    if not messages:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS:
                    return True
        elif role == "tool":
            name = str(msg.get("name") or "")
            if name in _TERMINAL_KANBAN_TOOLS:
                return True
    return False


def successful_current_kanban_terminal_transition(
    *,
    tool_calls: Iterable[Any] | None,
    messages: Iterable[dict] | None,
) -> Optional[str]:
    """Return the successful terminal tool from the current managed turn.

    A dispatcher worker does not need another provider request after its exact
    ``kanban_complete`` or ``kanban_block`` call has committed.  Match only
    tool results from *tool_calls* and require the result to bind the current
    task and run.  This prevents a stale result from an earlier turn, another
    card, or another run from terminating the conversation.
    """
    if not kanban_stop_nudge_enabled() or not tool_calls or not messages:
        return None

    task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    run_id_text = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    try:
        run_id = int(run_id_text)
    except (TypeError, ValueError):
        return None
    if run_id <= 0:
        return None

    current: dict[str, str] = {}
    for tool_call in tool_calls:
        name = _tool_call_name(tool_call)
        if name not in _TERMINAL_KANBAN_TOOLS:
            continue
        if isinstance(tool_call, dict):
            tool_call_id = str(tool_call.get("id") or "")
        else:
            tool_call_id = str(getattr(tool_call, "id", "") or "")
        if tool_call_id:
            current[tool_call_id] = name
    if not current:
        return None

    # Read newest-first and evaluate at most one result per current call id.
    # Providers may reuse a call id across turns; an older successful result
    # must never override the current call's failed result.
    checked: set[str] = set()
    for message in reversed(list(messages)):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        tool_call_id = str(message.get("tool_call_id") or "")
        expected_name = current.get(tool_call_id)
        if expected_name is None or str(message.get("name") or "") != expected_name:
            continue
        if tool_call_id in checked:
            continue
        checked.add(tool_call_id)
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            continue
        if payload.get("task_id") != task_id:
            continue
        if payload.get("run_id") != run_id:
            continue
        return expected_name
    return None


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Return a synthetic follow-up when a kanban worker exits without a terminal tool.

    Returns ``None`` when the guard should not fire (not a kanban worker,
    already completed/blocked, or nudge budget exhausted).
    """
    if not kanban_stop_nudge_enabled():
        return None
    if attempts >= max_attempts:
        return None
    if session_called_kanban_terminal(messages):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` is still `running`. Ending now without a board tool "
        "causes a protocol violation (clean exit with no "
        "`kanban_complete` / `kanban_block`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, OR `kanban_block(reason=...)` if you are blocked.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = [
    "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
    "successful_current_kanban_terminal_transition",
]
