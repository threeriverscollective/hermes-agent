"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
    successful_current_kanban_terminal_transition,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch






def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge
    assert "protocol violation" in nudge.lower() or "protocol" in nudge.lower()


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_current_successful_terminal_result_binds_exact_task_and_run(
    clear_kanban_env,
):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "42")
    tool_calls = [
        {
            "id": "call-current",
            "type": "function",
            "function": {"name": "kanban_complete", "arguments": "{}"},
        }
    ]
    messages = [
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "call-current",
            "content": '{"ok": true, "task_id": "t_abc", "run_id": 42}',
        }
    ]

    assert successful_current_kanban_terminal_transition(
        tool_calls=tool_calls,
        messages=messages,
    ) == "kanban_complete"


@pytest.mark.parametrize(
    "content",
    [
        '{"error": "not completed"}',
        '{"ok": true, "task_id": "t_other", "run_id": 42}',
        '{"ok": true, "task_id": "t_abc", "run_id": 41}',
    ],
)
def test_terminal_result_rejects_error_or_cross_authority_result(
    clear_kanban_env,
    content,
):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "42")
    tool_calls = [
        {
            "id": "call-current",
            "type": "function",
            "function": {"name": "kanban_complete", "arguments": "{}"},
        }
    ]
    messages = [
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "call-current",
            "content": content,
        }
    ]

    assert successful_current_kanban_terminal_transition(
        tool_calls=tool_calls,
        messages=messages,
    ) is None


def test_terminal_result_rejects_stale_tool_call_id(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "42")
    tool_calls = [
        {
            "id": "call-current",
            "type": "function",
            "function": {"name": "kanban_complete", "arguments": "{}"},
        }
    ]
    messages = [
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "call-stale",
            "content": '{"ok": true, "task_id": "t_abc", "run_id": 42}',
        }
    ]

    assert successful_current_kanban_terminal_transition(
        tool_calls=tool_calls,
        messages=messages,
    ) is None


def test_current_failure_wins_over_reused_stale_success(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "42")
    tool_calls = [
        {
            "id": "call-reused",
            "type": "function",
            "function": {"name": "kanban_complete", "arguments": "{}"},
        }
    ]
    messages = [
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "call-reused",
            "content": '{"ok": true, "task_id": "t_abc", "run_id": 42}',
        },
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "call-reused",
            "content": '{"error": "current handoff rejected"}',
        },
    ]

    assert successful_current_kanban_terminal_transition(
        tool_calls=tool_calls,
        messages=messages,
    ) is None






# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.


