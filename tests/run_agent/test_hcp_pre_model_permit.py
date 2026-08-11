"""Provider egress is impossible before the HCP permit guard allows it."""

from __future__ import annotations

from dataclasses import replace
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _response(text: str = "ok") -> SimpleNamespace:
    message = SimpleNamespace(content=text, tool_calls=None)
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "test/model",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    }
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=SimpleNamespace(
            prompt_tokens=7,
            completion_tokens=3,
            total_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2),
        ),
        model_dump=lambda **_: payload,
    )


def _codex_response(text: str = "ok") -> SimpleNamespace:
    payload = {
        "id": "resp-test",
        "object": "response",
        "model": "gpt-5.6-sol",
        "output": [{"type": "message", "content": text}],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 5,
            "total_tokens": 16,
            "input_tokens_details": {"cached_tokens": 4},
        },
    }
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=5,
            total_tokens=16,
            input_tokens_details=SimpleNamespace(cached_tokens=4),
        ),
        model_dump=lambda **_: payload,
        output=payload["output"],
    )


def _authorization(request_sha256: str):
    from hermes_cli.provider_request_guard import ProviderRequestAuthorization

    return ProviderRequestAuthorization(
        authorization_id="test-only-authorization",
        request_sha256=request_sha256,
        subject_sha256="sha256:" + "2" * 64,
        expires_at_monotonic=time.monotonic() + 30,
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
        default_headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer test-only-key",
        },
        default_query={},
        max_retries=0,
        _client=SimpleNamespace(follow_redirects=False),
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: callback(kwargs))
        ),
        close=lambda: None,
    )


def _codex_provider_client(callback, *, account_id: str | None = None):
    default_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer test-only-key",
        "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)",
        "originator": "codex_cli_rs",
    }
    if account_id is not None:
        default_headers["ChatGPT-Account-ID"] = account_id
    return SimpleNamespace(
        base_url="https://chatgpt.com/backend-api/codex/",
        default_headers=default_headers,
        default_query={},
        max_retries=0,
        _client=SimpleNamespace(follow_redirects=False),
        responses=SimpleNamespace(create=lambda **kwargs: callback(kwargs)),
        close=lambda: None,
    )


class _CompletingGuard:
    def __init__(self, *, events=None, completion_error: Exception | None = None):
        self.events = events
        self.completion_error = completion_error
        self.authorizations = 0
        self.results = []

    def __call__(self, **kwargs):
        from hermes_cli.provider_request_guard import ProviderRequestAuthorization

        self.authorizations += 1
        if self.events is not None:
            self.events.append("permit")
        return ProviderRequestAuthorization(
            authorization_id=f"authorization-{self.authorizations}",
            request_sha256=kwargs["request_sha256"],
            subject_sha256="sha256:" + "6" * 64,
            expires_at_monotonic=time.monotonic() + 30,
        )

    def complete_provider_attempt(self, *, result):
        from hermes_cli.provider_request_guard import ProviderAttemptAcknowledgment

        if self.events is not None:
            self.events.append("completion")
        self.results.append(result)
        if self.completion_error is not None:
            raise self.completion_error
        return ProviderAttemptAcknowledgment(
            authorization_id=result.authorization_id,
            request_sha256=result.request_sha256,
            subject_sha256=result.subject_sha256,
            usage_receipt_sha256="sha256:" + "9" * 64,
        )


class _CallbackCompletingGuard(_CompletingGuard):
    def __init__(self, callback, *, events=None):
        super().__init__(events=events)
        self.callback = callback

    def __call__(self, **kwargs):
        return self.callback(**kwargs)


def _codex_target() -> SimpleNamespace:
    return SimpleNamespace(
        api_mode="codex_responses",
        provider="openai-codex",
        api_key="test-only-key",
        base_url="https://chatgpt.com/backend-api/codex",
        _client_kwargs={
            "default_headers": {
                "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)",
                "originator": "codex_cli_rs",
            }
        },
        _provider_request_guard_context={
            "task_id": "card-7",
            "turn_id": "turn-1",
            "api_request_id": "turn-1:api:0",
            "session_id": "session-9",
            "profile_id": "hcp-general-implementer",
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
            "api_mode": "codex_responses",
            "api_call_count": 0,
        },
        _run_codex_stream=MagicMock(),
    )


def _codex_request() -> dict[str, object]:
    return {
        "model": "gpt-5.6-sol",
        "instructions": "Do the bounded task.",
        "input": [{"role": "user", "content": "exact"}],
        "store": False,
        "reasoning": {"effort": "high", "summary": "auto"},
        "max_output_tokens": 512,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "pck_exact",
        "extra_headers": {
            "session_id": "session-9",
            "x-client-request-id": "session-9",
        },
        "timeout": 30.0,
    }


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


def test_required_codex_guard_absence_blocks_before_provider(monkeypatch) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    provider = MagicMock()
    target = _codex_target()
    client = _codex_provider_client(provider)

    with pytest.raises(ProviderRequestBlocked, match="GUARD_UNAVAILABLE"):
        _dispatch_nonstreaming_api_request(
            target,
            _codex_request(),
            make_client=lambda _reason: client,
        )

    provider.assert_not_called()
    target._run_codex_stream.assert_not_called()


def test_guard_without_required_completion_method_blocks_before_provider(
    monkeypatch,
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    manager._provider_request_guard = lambda **kwargs: _authorization(
        kwargs["request_sha256"]
    )
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    provider = MagicMock(return_value=_codex_response())

    with pytest.raises(ProviderRequestBlocked, match="COMPLETION_UNAVAILABLE"):
        _dispatch_nonstreaming_api_request(
            _codex_target(),
            _codex_request(),
            make_client=lambda _reason: _codex_provider_client(provider),
        )

    provider.assert_not_called()


def test_codex_guard_binds_exact_route_immediately_before_provider(
    monkeypatch,
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestAuthorization

    events: list[str] = []
    observed: dict[str, object] = {}
    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True

    def allow(**kwargs):
        observed.update(kwargs)
        events.append("permit")
        return ProviderRequestAuthorization(
            authorization_id="codex-authorization",
            request_sha256=kwargs["request_sha256"],
            subject_sha256="sha256:" + "6" * 64,
            expires_at_monotonic=time.monotonic() + 30,
        )

    def provider(kwargs):
        events.append("provider")
        return _codex_response()

    manager._provider_request_guard = _CallbackCompletingGuard(allow, events=events)
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    target = _codex_target()
    client = _codex_provider_client(provider)

    def make_client(_reason):
        events.append("client")
        return client

    transported = _dispatch_nonstreaming_api_request(
        target,
        _codex_request(),
        make_client=make_client,
    )

    assert observed["request"] == {
        "include": ["reasoning.encrypted_content"],
        "input": [{"role": "user", "content": "exact"}],
        "instructions": "Do the bounded task.",
        "max_output_tokens": 512,
        "model": "gpt-5.6-sol",
        "prompt_cache_key": "pck_exact",
        "reasoning": {"effort": "high", "summary": "auto"},
        "store": False,
    }
    assert observed["provider"] == "openai-codex"
    assert observed["model"] == "gpt-5.6-sol"
    assert observed["api_mode"] == "codex_responses"
    assert observed["endpoint_origin"] == "https://chatgpt.com"
    assert observed["model_tokens_requested"] == 512
    assert str(observed["transport_identity_sha256"]).startswith("sha256:")
    assert transported.output
    assert events == ["client", "permit", "provider", "completion"]
    target._run_codex_stream.assert_not_called()


def test_codex_guard_requires_a_provider_side_output_ceiling(monkeypatch) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    manager._provider_request_guard = MagicMock()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    target = _codex_target()
    provider = MagicMock()
    request = _codex_request()
    request.pop("max_output_tokens")

    with pytest.raises(ProviderRequestBlocked, match="MODEL_BUDGET_UNBOUNDED"):
        _dispatch_nonstreaming_api_request(
            target,
            request,
            make_client=lambda _reason: _codex_provider_client(provider),
        )

    manager._provider_request_guard.assert_not_called()
    provider.assert_not_called()


@pytest.mark.parametrize(
    "extra_body",
    [
        {"stream": True},
        {"store": True},
        {"metadata": {"unbound": "value"}},
    ],
)
def test_codex_guard_rejects_extra_body_before_permit_or_provider(
    monkeypatch, extra_body: dict[str, object]
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    manager._provider_request_guard = MagicMock()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    target = _codex_target()
    provider = MagicMock()
    request = _codex_request()
    request["extra_body"] = extra_body

    with pytest.raises(ProviderRequestBlocked, match="ROUTE_UNSUPPORTED"):
        _dispatch_nonstreaming_api_request(
            target,
            request,
            make_client=lambda _reason: _codex_provider_client(provider),
        )

    manager._provider_request_guard.assert_not_called()
    provider.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [("store", True), ("store", None), ("stream", True)],
)
def test_codex_guard_requires_non_streaming_non_stored_final_body(
    monkeypatch, field: str, value: object
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    manager._provider_request_guard = MagicMock()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    target = _codex_target()
    provider = MagicMock()
    request = _codex_request()
    request[field] = value

    with pytest.raises(ProviderRequestBlocked, match="ROUTE_UNSUPPORTED"):
        _dispatch_nonstreaming_api_request(
            target,
            request,
            make_client=lambda _reason: _codex_provider_client(provider),
        )

    manager._provider_request_guard.assert_not_called()
    provider.assert_not_called()


def test_codex_guard_rejects_unbound_transport_header_before_provider(
    monkeypatch,
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    manager._provider_request_guard = MagicMock()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    target = _codex_target()
    provider = MagicMock()
    request = _codex_request()
    request["extra_headers"] = {
        "session_id": "session-9",
        "x-client-request-id": "session-9",
        "x-unbound-route": "forbidden",
    }

    with pytest.raises(ProviderRequestBlocked, match="TRANSPORT_UNSUPPORTED"):
        _dispatch_nonstreaming_api_request(
            target,
            request,
            make_client=lambda _reason: _codex_provider_client(provider),
        )

    manager._provider_request_guard.assert_not_called()
    provider.assert_not_called()


def test_codex_guard_binds_real_cloudflare_and_account_headers(monkeypatch) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins

    account_id = "account-bound-007c"
    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    authorize = MagicMock(
        side_effect=lambda **kwargs: _authorization(kwargs["request_sha256"])
    )
    guard = _CallbackCompletingGuard(authorize)
    manager._provider_request_guard = guard
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    target = _codex_target()
    target._client_kwargs["default_headers"]["ChatGPT-Account-ID"] = account_id
    provider = MagicMock(return_value=_codex_response())

    _dispatch_nonstreaming_api_request(
        target,
        _codex_request(),
        make_client=lambda _reason: _codex_provider_client(
            provider, account_id=account_id
        ),
    )

    provider.assert_called_once()
    assert authorize.call_args.kwargs["transport_identity_sha256"].startswith("sha256:")
    assert len(guard.results) == 1


def test_codex_guard_rejects_cloudflare_header_rebinding(monkeypatch) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    manager._provider_request_guard = MagicMock()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    target = _codex_target()
    provider = MagicMock()
    client = _codex_provider_client(provider)
    client.default_headers["originator"] = "codex_vscode"

    with pytest.raises(ProviderRequestBlocked, match="TRANSPORT_UNSUPPORTED"):
        _dispatch_nonstreaming_api_request(
            target,
            _codex_request(),
            make_client=lambda _reason: client,
        )

    manager._provider_request_guard.assert_not_called()
    provider.assert_not_called()


def test_codex_provider_retry_requires_a_fresh_authorization(monkeypatch) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestAuthorization

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    authorizations: list[str] = []

    def allow(**kwargs):
        authorization_id = f"codex-authorization-{len(authorizations) + 1}"
        authorizations.append(authorization_id)
        return ProviderRequestAuthorization(
            authorization_id=authorization_id,
            request_sha256=kwargs["request_sha256"],
            subject_sha256="sha256:" + "6" * 64,
            expires_at_monotonic=time.monotonic() + 30,
        )

    provider_calls = 0

    def provider(kwargs):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            raise TimeoutError("provider outcome unknown")
        return _codex_response()

    manager._provider_request_guard = _CallbackCompletingGuard(allow)
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    target = _codex_target()
    client = _codex_provider_client(provider)

    with pytest.raises(TimeoutError, match="outcome unknown"):
        _dispatch_nonstreaming_api_request(
            target, _codex_request(), make_client=lambda _reason: client
        )
    _dispatch_nonstreaming_api_request(
        target, _codex_request(), make_client=lambda _reason: client
    )

    assert provider_calls == 2
    assert authorizations == ["codex-authorization-1", "codex-authorization-2"]


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

    manager._provider_request_guard = _CallbackCompletingGuard(allow, events=events)
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
    assert events == ["client", "permit", "provider", "completion"]


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

    manager._provider_request_guard = _CallbackCompletingGuard(allow)
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
        "api_request_id": "turn-1:api:0",
        "session_id": "session-9",
        "profile_id": "hcp-general-implementer",
        "provider": "openai",
        "model": "test-model",
        "api_mode": "chat_completions",
        "api_call_count": 0,
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
    else:
        client.default_headers["X-Unbound-Route"] = "unbound"
    agent._provider_request_guard_context = {
        "task_id": "card-7",
        "turn_id": "turn-1",
        "api_request_id": "turn-1:api:0",
        "session_id": "session-9",
        "profile_id": "hcp-general-implementer",
        "provider": "openai",
        "model": "test-model",
        "api_mode": "chat_completions",
        "api_call_count": 0,
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

    manager._provider_request_guard = _CallbackCompletingGuard(allow)
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

    manager._provider_request_guard = _CallbackCompletingGuard(allow, events=events)
    monkeypatch.setattr(plugins, "_plugin_manager", manager)

    class Completions:
        @staticmethod
        def create(**kwargs):
            events.append("provider")
            return _response()

    client = SimpleNamespace(
        base_url="https://example.invalid/v1/",
        default_headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer test-only-key",
        },
        default_query={},
        max_retries=0,
        _client=SimpleNamespace(follow_redirects=False),
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
            "api_request_id": "turn-1:api:0",
            "session_id": "session-9",
            "profile_id": "hcp-general-implementer",
            "provider": "openai",
            "model": "test-model",
            "api_mode": "chat_completions",
            "api_call_count": 0,
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
    assert transported.choices
    assert events == ["permit", "provider", "completion"]


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
        "api_request_id": "turn-1:api:0",
        "session_id": "session-9",
        "profile_id": "hcp-general-implementer",
        "provider": "openai",
        "model": "test-model",
        "api_mode": "chat_completions",
        "api_call_count": 0,
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
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 2},
                },
            },
        )

    manager._provider_request_guard = _CallbackCompletingGuard(allow)
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
            "api_request_id": "turn-1:api:0",
            "session_id": "session-9",
            "profile_id": "hcp-general-implementer",
            "provider": "openai",
            "model": "test-model",
            "api_mode": "chat_completions",
            "api_call_count": 0,
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
        "api_request_id": "turn-1:api:0",
        "session_id": "session-9",
        "profile_id": "hcp-general-implementer",
        "provider": "openai",
        "model": "test-model",
        "api_mode": "chat_completions",
        "api_call_count": 0,
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

    manager._provider_request_guard = _CallbackCompletingGuard(mutate)
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
            "api_request_id": "turn-1:api:0",
            "session_id": "session-9",
            "profile_id": "hcp-general-implementer",
            "provider": "openai",
            "model": "test-model",
            "api_mode": "chat_completions",
            "api_call_count": 0,
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


@pytest.mark.parametrize("redirect_state", [True, None])
def test_managed_client_must_prove_redirects_disabled_before_permit_or_provider(
    monkeypatch, redirect_state
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    guard = _CompletingGuard()
    manager._provider_request_guard = guard
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    provider = MagicMock(return_value=_codex_response())
    client = _codex_provider_client(provider)
    if redirect_state is None:
        client._client = SimpleNamespace()
    else:
        client._client.follow_redirects = redirect_state

    with pytest.raises(ProviderRequestBlocked, match="TRANSPORT_UNSUPPORTED"):
        _dispatch_nonstreaming_api_request(
            _codex_target(),
            _codex_request(),
            make_client=lambda _reason: client,
        )

    assert guard.authorizations == 0
    provider.assert_not_called()


def test_managed_client_must_prove_sdk_retries_disabled_before_permit_or_provider(
    monkeypatch,
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    guard = _CompletingGuard()
    manager._provider_request_guard = guard
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    provider = MagicMock(return_value=_codex_response())
    client = _codex_provider_client(provider)
    client.max_retries = 1

    with pytest.raises(ProviderRequestBlocked, match="TRANSPORT_UNSUPPORTED"):
        _dispatch_nonstreaming_api_request(
            _codex_target(),
            _codex_request(),
            make_client=lambda _reason: client,
        )

    assert guard.authorizations == 0
    provider.assert_not_called()


def test_one_authorization_id_can_reach_only_one_provider_create(monkeypatch) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import (
        ProviderRequestAuthorization,
        ProviderRequestBlocked,
    )

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True

    def authorize(**kwargs):
        return ProviderRequestAuthorization(
            authorization_id="same-permit-id",
            request_sha256=kwargs["request_sha256"],
            subject_sha256="sha256:" + "6" * 64,
            expires_at_monotonic=time.monotonic() + 30,
        )

    guard = _CallbackCompletingGuard(authorize)
    manager._provider_request_guard = guard
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    provider = MagicMock(return_value=_codex_response())
    client = _codex_provider_client(provider)

    _dispatch_nonstreaming_api_request(
        _codex_target(), _codex_request(), make_client=lambda _reason: client
    )
    with pytest.raises(ProviderRequestBlocked, match="AUTHORIZATION_REPLAY"):
        _dispatch_nonstreaming_api_request(
            _codex_target(), _codex_request(), make_client=lambda _reason: client
        )

    provider.assert_called_once()


def test_success_is_completed_with_exact_usage_and_output_before_return(
    monkeypatch,
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins

    events: list[str] = []
    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    guard = _CompletingGuard(events=events)
    manager._provider_request_guard = guard
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    response = _codex_response("bounded")

    def provider(_kwargs):
        events.append("provider")
        return response

    result = _dispatch_nonstreaming_api_request(
        _codex_target(),
        _codex_request(),
        make_client=lambda _reason: _codex_provider_client(provider),
    )

    assert result is response
    assert events == ["permit", "provider", "completion"]
    attempt = guard.results[0]
    assert attempt.outcome == "SUCCESS"
    assert (attempt.input_tokens, attempt.output_tokens) == (11, 5)
    assert (attempt.cache_tokens, attempt.total_tokens) == (4, 16)
    assert attempt.wall_time_ms >= 0
    assert attempt.stall_time_ms == attempt.wall_time_ms
    expected = json.dumps(
        response.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert attempt.output_bytes == len(expected)


def test_ambiguous_success_usage_blocks_before_response_can_be_consumed(
    monkeypatch,
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    guard = _CompletingGuard()
    manager._provider_request_guard = guard
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    response = _codex_response()
    response.usage.total_tokens = None
    provider = MagicMock(return_value=response)

    with pytest.raises(ProviderRequestBlocked, match="USAGE_INVALID"):
        _dispatch_nonstreaming_api_request(
            _codex_target(),
            _codex_request(),
            make_client=lambda _reason: _codex_provider_client(provider),
        )

    provider.assert_called_once()
    assert len(guard.results) == 1
    assert guard.results[0].outcome == "PROVIDER_ERROR"
    assert guard.results[0].error_code == "PROVIDER_RESPONSE_INVALID"


def test_provider_exception_is_recorded_before_the_exception_can_retry(
    monkeypatch,
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins

    events: list[str] = []
    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    guard = _CompletingGuard(events=events)
    manager._provider_request_guard = guard
    monkeypatch.setattr(plugins, "_plugin_manager", manager)

    def provider(_kwargs):
        events.append("provider")
        raise TimeoutError("provider outcome unknown")

    with pytest.raises(TimeoutError, match="outcome unknown"):
        _dispatch_nonstreaming_api_request(
            _codex_target(),
            _codex_request(),
            make_client=lambda _reason: _codex_provider_client(provider),
        )

    assert events == ["permit", "provider", "completion"]
    attempt = guard.results[0]
    assert attempt.outcome == "PROVIDER_ERROR"
    assert attempt.input_tokens is None
    assert attempt.output_bytes == 0
    assert attempt.error_code == "PROVIDER_TIMEOUTERROR"


def test_completion_failure_blocks_a_successful_provider_response(monkeypatch) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    guard = _CompletingGuard(completion_error=OSError("HCP unavailable"))
    manager._provider_request_guard = guard
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    provider = MagicMock(return_value=_codex_response())

    with pytest.raises(ProviderRequestBlocked, match="COMPLETION_FAILED"):
        _dispatch_nonstreaming_api_request(
            _codex_target(),
            _codex_request(),
            make_client=lambda _reason: _codex_provider_client(provider),
        )

    provider.assert_called_once()


def test_completion_failure_blocks_provider_exception_from_reaching_retry(
    monkeypatch,
) -> None:
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from hermes_cli import plugins
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    manager = plugins.PluginManager()
    manager._provider_request_guard_required = True
    guard = _CompletingGuard(completion_error=OSError("HCP unavailable"))
    manager._provider_request_guard = guard
    monkeypatch.setattr(plugins, "_plugin_manager", manager)

    def provider(_kwargs):
        raise TimeoutError("must not reach retry")

    with pytest.raises(ProviderRequestBlocked, match="COMPLETION_FAILED"):
        _dispatch_nonstreaming_api_request(
            _codex_target(),
            _codex_request(),
            make_client=lambda _reason: _codex_provider_client(provider),
        )

    assert len(guard.results) == 1
    assert guard.results[0].outcome == "PROVIDER_ERROR"


def test_unknown_completion_exchange_can_retry_only_the_same_consumed_attempt() -> None:
    from hermes_cli.plugins import PluginManager
    from hermes_cli.provider_request_guard import (
        ProviderAttemptAcknowledgment,
        ProviderAttemptResult,
        ProviderRequestBlocked,
    )

    class _RetryingCompletion:
        def __init__(self) -> None:
            self.calls = 0

        def complete_provider_attempt(self, *, result):
            self.calls += 1
            if self.calls == 1:
                raise OSError("ack outcome unknown")
            return ProviderAttemptAcknowledgment(
                authorization_id=result.authorization_id,
                request_sha256=result.request_sha256,
                subject_sha256=result.subject_sha256,
                usage_receipt_sha256="sha256:" + "9" * 64,
            )

    attempt = ProviderAttemptResult(
        authorization_id="authorization-1",
        request_sha256="sha256:" + "1" * 64,
        subject_sha256="sha256:" + "2" * 64,
        outcome="PROVIDER_ERROR",
        input_tokens=None,
        output_tokens=None,
        cache_tokens=None,
        total_tokens=None,
        wall_time_ms=17,
        stall_time_ms=17,
        output_bytes=0,
        error_code="PROVIDER_TIMEOUTERROR",
    )
    manager = PluginManager()
    guard = _RetryingCompletion()
    manager._provider_request_guard = guard
    manager._provider_request_authorizations.add(attempt.authorization_id)

    with pytest.raises(ProviderRequestBlocked, match="COMPLETION_FAILED"):
        manager.complete_provider_request_guard(result=attempt)
    changed = replace(attempt, wall_time_ms=18, stall_time_ms=18)
    with pytest.raises(ProviderRequestBlocked, match="COMPLETION_MISMATCH"):
        manager.complete_provider_request_guard(result=changed)

    acknowledged = manager.complete_provider_request_guard(result=attempt)

    assert acknowledged.authorization_id == attempt.authorization_id
    assert guard.calls == 2
    with pytest.raises(ProviderRequestBlocked, match="COMPLETION_REPLAY"):
        manager.complete_provider_request_guard(result=attempt)
