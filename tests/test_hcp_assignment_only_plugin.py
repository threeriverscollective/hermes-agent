"""Contract for the provider-free HCP assignment command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest


def _request(*profiles: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "card_id": "card:007c:implementer-a",
        "normalized_title": "Implement the admitted card",
        "capsule_summary": "Use only the closed admitted capsule.",
        "allowed_profile_ids": list(profiles),
        "required_capability_classes": ["repository-read"],
        "repository_policy_id": "repository-policy:007c",
        "routing_policy_id": "routing-policy:single-allowed-profile-v1",
    }


def _parser(module) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    module._register_cli(parser)
    return parser


def test_single_installed_allowlisted_profile_is_selected_without_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli.hcp_assignment import route_assignment

    root = tmp_path / "profiles"
    (root / "hcp-general-implementer").mkdir(parents=True)
    provider_calls: list[object] = []
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda *args, **kwargs: provider_calls.append((args, kwargs)),
    )

    result = route_assignment(
        _request("hcp-general-implementer"),
        profiles_root=root,
        run_id="assignment:007c:a",
        provider="openai-codex",
        model="gpt-5.6-sol",
        effort="high",
        attempted_profile_id="hcp_assignment_only",
    )

    assert result["selection"] == "hcp-general-implementer"
    assert result["usage"] == {
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "effort": "high",
        "attempted_profile_id": "hcp_assignment_only",
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_tokens": 0,
        "total_tokens": 0,
        "cost_microunits": 0,
        "context_bytes": len(
            json.dumps(
                _request("hcp-general-implementer"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ),
        "tool_bytes": 0,
        "wall_time_ms": 0,
        "queue_time_ms": 0,
        "retry_count": 0,
        "termination_reason": "completed",
        "broker_api_calls": 0,
        "quota_units": 0,
    }
    assert provider_calls == []


@pytest.mark.parametrize(
    "profiles",
    [
        (),
        ("hcp-general-implementer", "hcp-pr-acceptance-reviewer"),
        ("missing-profile",),
    ],
)
def test_ambiguous_or_unavailable_profile_returns_no_fit_without_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profiles: tuple[str, ...],
) -> None:
    from hermes_cli.hcp_assignment import route_assignment

    root = tmp_path / "profiles"
    (root / "hcp-general-implementer").mkdir(parents=True)
    (root / "hcp-pr-acceptance-reviewer").mkdir()
    provider_calls: list[object] = []
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda *args, **kwargs: provider_calls.append((args, kwargs)),
    )

    result = route_assignment(
        _request(*profiles),
        profiles_root=root,
        run_id="assignment:007c:hold",
        provider="openai-codex",
        model="gpt-5.6-sol",
        effort="high",
        attempted_profile_id="hcp_assignment_only",
    )

    assert result["selection"] == "NO_FIT"
    assert result["usage"]["termination_reason"] == "no_fit"
    assert provider_calls == []


def test_cli_registration_is_closed_and_requires_a_bound_profiles_root() -> None:
    from hermes_cli import hcp_assignment as module

    parsed = _parser(module).parse_args(
        [
            "--input-json",
            "--run-id",
            "assignment:007c:a",
            "--provider",
            "openai-codex",
            "--model",
            "gpt-5.6-sol",
            "--effort",
            "high",
            "--attempted-profile-id",
            "hcp_assignment_only",
            "--profiles-root",
            "/private/var/hcp-007c/profiles",
        ]
    )

    assert parsed.profiles_root == "/private/var/hcp-007c/profiles"


def test_plugin_registers_only_the_assignment_cli_command() -> None:
    from hermes_cli import hcp_assignment as module

    registrations: list[dict[str, object]] = []

    class Context:
        def register_cli_command(self, **kwargs: object) -> None:
            registrations.append(kwargs)

        def register_auxiliary_task(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("assignment must not register a model route")

    module.register(Context())

    assert [item["name"] for item in registrations] == ["hcp-assignment"]
