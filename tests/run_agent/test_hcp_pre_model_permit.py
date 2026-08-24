"""Provider egress is impossible before the HCP permit guard allows it."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _deliver_manually_injected_plugin_manager(monkeypatch, tmp_path) -> None:
    """Keep these unit tests isolated from unrelated profile discovery.

    The current plugin manager lazily discovers on first delivery.  These
    tests install the guard (or its deliberate absence) directly, so mark the
    injected manager as already discovered before exercising that exact unit
    boundary.
    """

    from hermes_cli import plugins

    ledger_home = tmp_path / "provider-ledger-home"
    ledger_home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(ledger_home))

    def delivery_manager():
        manager = plugins.get_plugin_manager()
        manager._discovered = True
        return manager

    monkeypatch.setattr(plugins, "_delivery_manager", delivery_manager)


def _response(text: str = "ok") -> SimpleNamespace:
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


@pytest.fixture()
def agent() -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        value = AIAgent(
            api_key="test-only-key",
            base_url="https://example.invalid/v1",
            provider="openai",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    value._cached_system_prompt = "You are helpful."
    value.max_tokens = 64
    value._use_prompt_caching = False
    value.tool_delay = 0
    value.compression_enabled = False
    value.save_trajectories = False
    return value


def _run(agent: AIAgent):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("hello")


def _provider_client(callback):
    return SimpleNamespace(
        base_url="https://example.invalid/v1/",
        max_retries=0,
        default_headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer test-only-key",
        },
        default_query={},
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: callback(kwargs))
        ),
        close=lambda: None,
    )


def test_required_guard_absence_blocks_before_provider(agent, monkeypatch) -> None:
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    provider = MagicMock(return_value=_response())
    client = _provider_client(provider)
    monkeypatch.setattr(
        agent, "_create_request_openai_client", lambda **_kwargs: client
    )

    with pytest.raises(ProviderRequestBlocked, match="UNAVAILABLE"):
        _run(agent)

    provider.assert_not_called()


def test_required_guard_refusal_returns_only_a_neutral_hold(agent, monkeypatch) -> None:
    from hermes_cli import kanban_db, plugins

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    expected = {
        "final_response": "",
        "messages": [],
        "api_calls": 0,
        "completed": False,
        "failed": False,
        "neutral_hold": True,
        "failure_reason": "neutral_hold",
        "error": "PROVIDER_REQUEST_GUARD_UNAVAILABLE",
    }
    hold = MagicMock(return_value=expected)
    monkeypatch.setattr(kanban_db, "record_provider_guard_hold", hold)
    provider = MagicMock(return_value=_response())
    client = _provider_client(provider)
    monkeypatch.setattr(
        agent, "_create_request_openai_client", lambda **_kwargs: client
    )

    assert _run(agent) == expected
    provider.assert_not_called()
    hold.assert_called_once_with("PROVIDER_REQUEST_GUARD_UNAVAILABLE")


def test_required_guard_rejects_preflight_model_routes(agent, monkeypatch) -> None:
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    agent.compression_enabled = True
    compressor = MagicMock()
    agent._compress_context = compressor
    provider = MagicMock(return_value=_response())
    agent._interruptible_api_call = provider

    with pytest.raises(ProviderRequestBlocked, match="ROUTE_UNSUPPORTED"):
        _run(agent)

    compressor.assert_not_called()
    provider.assert_not_called()


def test_required_guard_rejects_goal_mode_before_any_model(agent, monkeypatch) -> None:
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    provider = MagicMock(return_value=_response())
    agent._interruptible_api_call = provider

    with pytest.raises(ProviderRequestBlocked, match="ROUTE_UNSUPPORTED"):
        _run(agent)

    provider.assert_not_called()


@pytest.mark.parametrize(
    ("api_mode", "provider"),
    [
        ("codex_responses", "openai"),
        ("anthropic_messages", "anthropic"),
        ("bedrock_converse", "bedrock"),
        ("chat_completions", "moa"),
    ],
)
def test_guard_rechecks_route_after_in_turn_fallback(
    agent, monkeypatch, api_mode: str, provider: str
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    agent.api_mode = api_mode
    agent.provider = provider
    agent._run_codex_stream = MagicMock()
    agent._anthropic_messages_create = MagicMock()
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock()))
    )
    make_client = MagicMock()

    with pytest.raises(ProviderRequestBlocked, match="ROUTE_UNSUPPORTED"):
        _dispatch_nonstreaming_api_request(agent, {}, make_client=make_client)

    make_client.assert_not_called()
    agent._run_codex_stream.assert_not_called()
    agent._anthropic_messages_create.assert_not_called()
    agent.client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("provider", ["openai-codex", "xai-oauth"])
def test_guarded_codex_responses_route_is_explicitly_supported(provider) -> None:
    from hermes_cli.provider_request_guard import (
        provider_request_guard_route_supported,
    )

    assert provider_request_guard_route_supported(
        api_mode="codex_responses",
        provider=provider,
    ) is True


def test_guarded_xai_codex_response_is_permitted_immediately_before_provider(
    monkeypatch,
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestAuthorization

    events: list[str] = []
    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True

    def allow(**kwargs):
        events.append("permit")
        assert kwargs["provider"] == "xai-oauth"
        assert kwargs["api_mode"] == "codex_responses"
        assert kwargs["transport_mode"] == "non_streaming"
        assert kwargs["model_tokens_requested"] == 73
        return ProviderRequestAuthorization(
            authorization_id="test-only-xai-codex-authorization",
            request_sha256=kwargs["request_sha256"],
            subject_sha256="sha256:" + "6" * 64,
            expires_at_monotonic=time.monotonic() + 30,
        )

    manager._provider_request_guard = allow
    monkeypatch.setattr(plugins, "_plugin_manager", manager)

    class Responses:
        @staticmethod
        def create(**kwargs):
            events.append("provider")
            assert kwargs["model"] == "grok-4.6"
            assert kwargs["max_output_tokens"] == 73
            assert "stream" not in kwargs
            return SimpleNamespace(id="response-1", output=[])

    client = SimpleNamespace(
        base_url="https://api.x.ai/v1/",
        max_retries=0,
        default_headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer test-only-key",
        },
        default_query={},
        responses=Responses(),
    )
    target = SimpleNamespace(
        api_mode="codex_responses",
        provider="xai-oauth",
        model="grok-4.6",
        api_key="test-only-key",
        base_url="https://api.x.ai/v1",
        _provider_request_guard_context={
            "task_id": "card-7",
            "turn_id": "turn-1",
            "api_request_id": "turn-1:api:1",
            "session_id": "session-9",
            "profile_id": "implementer",
            "provider": "xai-oauth",
            "model": "grok-4.6",
            "api_mode": "codex_responses",
            "api_call_count": 1,
        },
    )

    result = _dispatch_nonstreaming_api_request(
        target,
        {
            "model": "grok-4.6",
            "input": "bounded",
            "max_output_tokens": 73,
        },
        make_client=lambda _reason: client,
    )

    assert result.id == "response-1"
    assert events == ["permit", "provider"]


def test_guarded_codex_response_rejects_streaming_from_extra_body(
    monkeypatch,
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    manager._provider_request_guard = MagicMock()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    provider = MagicMock()
    client = SimpleNamespace(
        base_url="https://api.x.ai/v1/",
        max_retries=0,
        default_headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer test-only-key",
        },
        default_query={},
        responses=SimpleNamespace(create=provider),
    )
    target = SimpleNamespace(
        api_mode="codex_responses",
        provider="xai-oauth",
        model="grok-4.6",
        api_key="test-only-key",
        base_url="https://api.x.ai/v1",
        _provider_request_guard_context={
            "task_id": "card-7",
            "turn_id": "turn-1",
            "api_request_id": "turn-1:api:1",
            "session_id": "session-9",
            "profile_id": "implementer",
            "provider": "xai-oauth",
            "model": "grok-4.6",
            "api_mode": "codex_responses",
            "api_call_count": 1,
        },
    )

    with pytest.raises(ProviderRequestBlocked, match="ROUTE_UNSUPPORTED"):
        _dispatch_nonstreaming_api_request(
            target,
            {
                "model": "grok-4.6",
                "input": "bounded",
                "max_output_tokens": 73,
                "extra_body": {"stream": True},
            },
            make_client=lambda _reason: client,
        )

    manager._provider_request_guard.assert_not_called()
    provider.assert_not_called()


def test_guard_is_immediately_before_provider(agent, monkeypatch) -> None:
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestAuthorization

    events: list[str] = []
    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True

    def allow(**kwargs):
        events.append("permit")
        return ProviderRequestAuthorization(
            authorization_id="test-only-authorization",
            request_sha256=kwargs["request_sha256"],
            subject_sha256="sha256:" + "6" * 64,
            expires_at_monotonic=time.monotonic() + 30,
        )

    manager._provider_request_guard = allow
    monkeypatch.setattr(plugins, "_plugin_manager", manager)

    def provider(_request):
        events.append("provider")
        return _response()

    client = _provider_client(provider)

    def make_client(**_kwargs):
        events.append("client")
        return client

    monkeypatch.setattr(agent, "_create_request_openai_client", make_client)
    result = _run(agent)

    assert result["completed"] is True
    assert events == ["client", "permit", "provider"]


def test_authorization_expiry_is_rechecked_at_provider_transport(
    agent, monkeypatch
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import (
        ProviderRequestAuthorization,
        ProviderRequestBlocked,
    )

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    events: list[str] = []

    def allow(**kwargs):
        events.append("permit")
        return ProviderRequestAuthorization(
            authorization_id="expires-before-transport",
            request_sha256=kwargs["request_sha256"],
            subject_sha256="sha256:" + "6" * 64,
            expires_at_monotonic=100.0,
        )

    manager._provider_request_guard = allow
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    ticks = iter((99.0, 101.0))
    monkeypatch.setattr(
        "hermes_cli.provider_request_guard.time.monotonic", lambda: next(ticks)
    )
    provider = MagicMock()
    client = _provider_client(provider)
    agent._provider_request_guard_context = {
        "task_id": "card-7",
        "turn_id": "turn-1",
        "api_request_id": "turn-1:api:1",
        "session_id": "session-9",
        "profile_id": "hcp-general-implementer",
        "provider": "openai",
        "model": "test-model",
        "api_mode": "chat_completions",
        "api_call_count": 1,
    }

    def make_client(_reason):
        events.append("client")
        return client

    with pytest.raises(ProviderRequestBlocked, match="AUTHORIZATION_EXPIRED"):
        _dispatch_nonstreaming_api_request(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "exact"}],
                "max_tokens": 37,
            },
            make_client=make_client,
        )

    assert events == ["client", "permit"]
    provider.assert_not_called()


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("base_url", "ENDPOINT_MISMATCH"),
        ("default_query", "TRANSPORT_UNSUPPORTED"),
        ("default_headers", "TRANSPORT_UNSUPPORTED"),
        ("max_retries", "TRANSPORT_UNSUPPORTED"),
    ],
)
def test_guard_refuses_unbound_client_transport_state(
    agent, monkeypatch, mutation, error
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    manager._provider_request_guard = MagicMock()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    provider = MagicMock()
    client = _provider_client(provider)
    if mutation == "base_url":
        client.base_url = "https://different.invalid/alternate/v1/"
    elif mutation == "default_query":
        client.default_query = {"api-version": "unbound"}
    elif mutation == "max_retries":
        client.max_retries = 1
    else:
        client.default_headers["X-Unbound-Route"] = "unbound"
    agent._provider_request_guard_context = {
        "task_id": "card-7",
        "turn_id": "turn-1",
        "api_request_id": "turn-1:api:1",
        "session_id": "session-9",
        "profile_id": "hcp-general-implementer",
        "provider": "openai",
        "model": "test-model",
        "api_mode": "chat_completions",
        "api_call_count": 1,
    }

    with pytest.raises(ProviderRequestBlocked, match=error):
        _dispatch_nonstreaming_api_request(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "exact"}],
                "max_tokens": 37,
            },
            make_client=lambda _reason: client,
        )

    manager._provider_request_guard.assert_not_called()
    provider.assert_not_called()


def test_provider_retry_requires_a_fresh_authorization(agent, monkeypatch) -> None:
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestAuthorization

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    authorizations: list[str] = []

    def allow(**kwargs):
        authorization_id = f"test-only-authorization-{len(authorizations) + 1}"
        authorizations.append(authorization_id)
        return ProviderRequestAuthorization(
            authorization_id=authorization_id,
            request_sha256=kwargs["request_sha256"],
            subject_sha256="sha256:" + "6" * 64,
            expires_at_monotonic=time.monotonic() + 30,
        )

    manager._provider_request_guard = allow
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    provider_calls = 0

    def provider(_request):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            raise TimeoutError("provider outcome unknown")
        return _response()

    client = _provider_client(provider)
    monkeypatch.setattr(
        agent, "_create_request_openai_client", lambda **_kwargs: client
    )
    result = _run(agent)

    assert result["completed"] is True
    assert provider_calls == 2
    assert authorizations == [
        "test-only-authorization-1",
        "test-only-authorization-2",
    ]


def test_iteration_limit_summary_never_bypasses_required_guard(
    agent, monkeypatch
) -> None:
    from hermes_cli import plugins

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    provider = MagicMock()
    agent._ensure_primary_openai_client = provider
    messages = [{"role": "user", "content": "hello"}]

    result = agent._handle_max_iterations(messages, 1)

    assert result == (
        f"I reached the maximum iterations ({agent.max_iterations}); "
        "the managed run ended without an additional model request."
    )
    assert messages[-1] == {"role": "assistant", "content": result}
    provider.assert_not_called()


def test_unconfigured_non_hcp_session_is_unchanged(agent, monkeypatch) -> None:
    from hermes_cli import plugins

    monkeypatch.setattr(plugins, "_plugin_manager", plugins.PluginManager())
    # This endpoint is valid for an ACP runtime but intentionally is not an
    # HTTP origin. An inactive guard must not parse or constrain it.
    agent.base_url = "acp://copilot"
    provider = MagicMock(return_value=_response("ordinary"))
    agent._interruptible_api_call = provider

    result = _run(agent)

    assert result["completed"] is True
    assert result["final_response"] == "ordinary"
    provider.assert_called_once()


def test_guard_receives_the_exact_final_model_payload(monkeypatch) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestAuthorization

    observed: dict[str, object] = {}
    events: list[str] = []
    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True

    def allow(**kwargs):
        observed.update(kwargs)
        events.append("permit")
        return ProviderRequestAuthorization(
            authorization_id="test-only-final-payload",
            request_sha256=kwargs["request_sha256"],
            subject_sha256="sha256:" + "6" * 64,
            expires_at_monotonic=time.monotonic() + 30,
        )

    manager._provider_request_guard = allow
    monkeypatch.setattr(plugins, "_plugin_manager", manager)

    class Completions:
        @staticmethod
        def create(**kwargs):
            events.append("provider")
            return kwargs

    client = SimpleNamespace(
        base_url="https://example.invalid/v1/",
        max_retries=0,
        default_headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer test-only-key",
        },
        default_query={},
        chat=SimpleNamespace(completions=Completions()),
    )
    target = SimpleNamespace(
        api_mode="chat_completions",
        provider="openai",
        api_key="test-only-key",
        base_url="https://example.invalid/v1",
        _provider_request_guard_context={
            "task_id": "card-7",
            "turn_id": "turn-1",
            "api_request_id": "turn-1:api:1",
            "session_id": "session-9",
            "profile_id": "hcp-general-implementer",
            "provider": "openai",
            "model": "test-model",
            "api_mode": "chat_completions",
            "api_call_count": 1,
        },
    )
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "exact"}],
        "max_tokens": 37,
        "timeout": 5,
    }

    transported = _dispatch_nonstreaming_api_request(
        target,
        payload,
        make_client=lambda _reason: client,
    )

    assert observed["request"] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "exact"}],
        "max_tokens": 37,
    }
    assert str(observed["transport_identity_sha256"]).startswith("sha256:")
    assert len(str(observed["transport_identity_sha256"])) == 71
    assert transported["timeout"] == 5
    assert events == ["permit", "provider"]


@pytest.mark.parametrize("control", ["extra_headers", "extra_query"])
def test_guard_rejects_unbound_request_transport_controls(
    agent, monkeypatch, control
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    manager._provider_request_guard = MagicMock()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    provider = MagicMock()
    client = _provider_client(provider)
    agent._provider_request_guard_context = {
        "task_id": "card-7",
        "turn_id": "turn-1",
        "api_request_id": "turn-1:api:1",
        "session_id": "session-9",
        "profile_id": "hcp-general-implementer",
        "provider": "openai",
        "model": "test-model",
        "api_mode": "chat_completions",
        "api_call_count": 1,
    }
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "exact"}],
        "max_tokens": 37,
        control: {"unbound": "value"},
    }

    with pytest.raises(ProviderRequestBlocked, match="TRANSPORT_UNSUPPORTED"):
        _dispatch_nonstreaming_api_request(
            agent, payload, make_client=lambda _reason: client
        )

    manager._provider_request_guard.assert_not_called()
    provider.assert_not_called()


def test_guard_binds_sdk_extra_body_to_the_actual_http_body(monkeypatch) -> None:
    import httpx
    from openai import OpenAI

    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestAuthorization

    observed: dict[str, object] = {}
    wire_bodies: list[dict[str, object]] = []
    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True

    def allow(**kwargs):
        observed.update(kwargs)
        return ProviderRequestAuthorization(
            authorization_id="test-only-extra-body",
            request_sha256=kwargs["request_sha256"],
            subject_sha256="sha256:" + "6" * 64,
            expires_at_monotonic=time.monotonic() + 30,
        )

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        wire_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "overridden-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    manager._provider_request_guard = allow
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAI(
        api_key="test-only-key",
        base_url="https://example.invalid/v1",
        http_client=http_client,
        max_retries=0,
    )
    target = SimpleNamespace(
        api_mode="chat_completions",
        provider="openai",
        api_key="test-only-key",
        base_url="https://example.invalid/v1",
        _provider_request_guard_context={
            "task_id": "card-7",
            "turn_id": "turn-1",
            "api_request_id": "turn-1:api:1",
            "session_id": "session-9",
            "profile_id": "hcp-general-implementer",
            "provider": "openai",
            "model": "test-model",
            "api_mode": "chat_completions",
            "api_call_count": 1,
        },
    )
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "base"}],
        "max_tokens": 37,
        "extra_body": {
            "model": "overridden-model",
            "messages": [{"role": "user", "content": "overridden"}],
            "temperature": 1.7,
        },
    }

    _dispatch_nonstreaming_api_request(
        target, payload, make_client=lambda _reason: client
    )
    client.close()

    assert observed["request"] == wire_bodies[0]
    assert observed["request"]["model"] == "overridden-model"
    assert observed["request"]["messages"][0]["content"] == "overridden"


def test_guard_rejects_a_request_without_an_output_token_ceiling(
    agent, monkeypatch
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    manager._provider_request_guard = MagicMock()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    provider = MagicMock()
    client = _provider_client(provider)
    agent._provider_request_guard_context = {
        "task_id": "card-7",
        "turn_id": "turn-1",
        "api_request_id": "turn-1:api:1",
        "session_id": "session-9",
        "profile_id": "hcp-general-implementer",
        "provider": "openai",
        "model": "test-model",
        "api_mode": "chat_completions",
        "api_call_count": 1,
    }

    with pytest.raises(ProviderRequestBlocked, match="MODEL_BUDGET_UNBOUNDED"):
        _dispatch_nonstreaming_api_request(
            agent,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "exact"}],
            },
            make_client=lambda _reason: client,
        )

    manager._provider_request_guard.assert_not_called()
    provider.assert_not_called()


def test_managed_worker_cannot_resolve_or_call_an_auxiliary_model(
    agent, monkeypatch
) -> None:
    from agent import auxiliary_client
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    resolver = MagicMock(side_effect=AssertionError("auxiliary resolution reached"))
    monkeypatch.setattr(auxiliary_client, "_resolve_task_provider_model", resolver)

    with pytest.raises(ProviderRequestBlocked, match="AUXILIARY_DISABLED"):
        auxiliary_client.call_llm(
            task="vision",
            messages=[{"role": "user", "content": "must not leave"}],
        )

    resolver.assert_not_called()

    with pytest.raises(ProviderRequestBlocked, match="AUXILIARY_DISABLED"):
        auxiliary_client._get_cached_client("auto")


def test_guard_cannot_mutate_the_authorized_request_before_transport(
    monkeypatch,
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import (
        ProviderRequestAuthorization,
        ProviderRequestBlocked,
    )

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True

    def mutate(**kwargs):
        kwargs["request"]["messages"][0]["content"] = "changed"
        return ProviderRequestAuthorization(
            authorization_id="test-only-mutated",
            request_sha256=kwargs["request_sha256"],
            subject_sha256="sha256:" + "6" * 64,
            expires_at_monotonic=time.monotonic() + 30,
        )

    manager._provider_request_guard = mutate
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    provider = MagicMock()
    client = _provider_client(provider)
    target = SimpleNamespace(
        api_mode="chat_completions",
        provider="openai",
        api_key="test-only-key",
        base_url="https://example.invalid/v1",
        _provider_request_guard_context={
            "task_id": "card-7",
            "turn_id": "turn-1",
            "api_request_id": "turn-1:api:1",
            "session_id": "session-9",
            "profile_id": "hcp-general-implementer",
            "provider": "openai",
            "model": "test-model",
            "api_mode": "chat_completions",
            "api_call_count": 1,
        },
    )
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "exact"}],
        "max_tokens": 37,
    }

    with pytest.raises(ProviderRequestBlocked, match="CHANGED"):
        _dispatch_nonstreaming_api_request(
            target, payload, make_client=lambda _reason: client
        )
    provider.assert_not_called()
    assert payload["messages"][0]["content"] == "exact"
