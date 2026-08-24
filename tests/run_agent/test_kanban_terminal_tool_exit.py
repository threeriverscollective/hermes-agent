"""Managed kanban terminal tools end a turn without another model call."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _tool_def(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "terminal kanban lifecycle transition",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_response(name: str, call_id: str) -> SimpleNamespace:
    call = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )
    message = SimpleNamespace(content="", tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(
        id="chatcmpl-kanban-terminal",
        choices=[choice],
        model="test/model",
        usage=None,
    )


def _agent(*, max_iterations: int = 1) -> AIAgent:
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[_tool_def("kanban_complete")],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are a managed kanban worker."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def test_successful_terminal_tool_ends_without_post_completion_provider_call(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_exact")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    agent = _agent()
    agent.client.chat.completions.create.side_effect = [
        _tool_response("kanban_complete", "call-terminal")
    ]

    with (
        patch(
            "run_agent.handle_function_call",
            return_value=json.dumps(
                {"ok": True, "task_id": "t_exact", "run_id": 42}
            ),
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("complete the assigned task")

    assert agent.client.chat.completions.create.call_count == 1
    assert result["completed"] is True
    assert result["turn_exit_reason"] == (
        "kanban_terminal_tool_success(kanban_complete)"
    )
    assert result["final_response"].endswith("kanban_complete.")


def test_failed_terminal_tool_still_allows_model_recovery(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_exact")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    agent = _agent(max_iterations=2)
    from tests.run_agent.test_run_agent import _mock_response

    agent.client.chat.completions.create.side_effect = [
        _tool_response("kanban_complete", "call-failed"),
        _mock_response(content="I will correct the failed handoff.", finish_reason="stop"),
    ]

    with (
        patch(
            "run_agent.handle_function_call",
            return_value=json.dumps({"error": "handoff rejected"}),
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("complete the assigned task")

    assert agent.client.chat.completions.create.call_count == 2
    assert result["turn_exit_reason"].startswith("text_response(")
