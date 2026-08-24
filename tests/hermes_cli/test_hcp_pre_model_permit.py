"""Contract tests for Hermes' HCP post-claim pre-model permit boundary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import socket
import struct
import threading
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _sign(key: Ed25519PrivateKey, value: object) -> str:
    return key.sign(_canonical(value)).hex()


def _read_frame(channel: socket.socket) -> dict[str, object]:
    header = channel.recv(4)
    assert len(header) == 4
    length = struct.unpack(">I", header)[0]
    payload = bytearray()
    while len(payload) < length:
        payload.extend(channel.recv(length - len(payload)))
    return json.loads(payload)


def _write_frame(channel: socket.socket, value: dict[str, object]) -> None:
    encoded = _canonical(value)
    channel.sendall(struct.pack(">I", len(encoded)) + encoded)


PEER_KEY = Ed25519PrivateKey.from_private_bytes(b"p" * 32)
SERVER_KEY = Ed25519PrivateKey.from_private_bytes(b"s" * 32)


def _snapshot() -> dict[str, object]:
    return {
        "schema_version": "hcp.pre-model-permit.v2",
        "card_id": "card-7",
        "bead_id": "hcp-rcv-007c",
        "item_revision": "revision-4",
        "repository_id": "repository-9",
        "protected_source_revision": "1" * 40,
        "authority_digest": "sha256:" + "2" * 64,
        "binding_digest": "sha256:" + "3" * 64,
        "scope_digest": "sha256:" + "4" * 64,
        "dependency_digest": "sha256:" + "5" * 64,
        "collision_digest": "sha256:" + "6" * 64,
        "revocation_digest": "sha256:" + "7" * 64,
        "run_id": "hcp-run-11",
        "generation_id": "generation-3",
        "selected_profile_id": "hcp-general-implementer",
        "allowed_profile_ids": ["hcp-general-implementer"],
        "required_capability_classes": ["implementation"],
        "profile_capability_classes": ["implementation"],
        "auth_mode": "API",
        "budget_enforcement_mode": "ENFORCED",
        "model_budget_remaining": 4096,
        "retry_budget_remaining": 2,
        "budget_policy_digest": "sha256:" + "8" * 64,
    }


def _manifest() -> dict[str, object]:
    return {
        "schema_version": "hermes.hcp.pre-model-permit-client.v2",
        "task_id": "card-7",
        "generation_id": "generation-3",
        "board_id": "hcp-board",
        "profile_id": "hcp-general-implementer",
        "binding_digest": "sha256:" + "3" * 64,
        "claim_lock_sha256": _digest_text("host:123"),
        "provider": "openai",
        "model": "test-model",
        "api_mode": "chat_completions",
        "endpoint_origin": "https://example.invalid",
        "peer_identity": "hermes-worker/card-7",
        "peer_key_id": "hermes-peer-key-1",
        "server_identity": "hcp-permit-server",
        "server_key_id": "hcp-server-key-1",
        "permit_ttl_seconds": 15,
        "model_tokens_per_request": 512,
        "reasoning_effort": None,
        "transport_mode": "non_streaming",
        "snapshot": _snapshot(),
    }


def _digest_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _socket_path(tmp_path: Path) -> Path:
    token = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    path = Path("/tmp") / f"hcp-{token}.sock"
    path.unlink(missing_ok=True)
    return path


def _context() -> dict[str, object]:
    return {
        "task_id": "card-7",
        "turn_id": "turn-1",
        "api_request_id": "turn-1:api:1",
        "session_id": "session-9",
        "profile_id": "hcp-general-implementer",
        "provider": "openai",
        "model": "test-model",
        "api_mode": "chat_completions",
        "endpoint_origin": "https://example.invalid",
        "transport_identity_sha256": "sha256:" + "d" * 64,
        "transport_mode": "non_streaming",
        "api_call_count": 1,
        "model_tokens_requested": 31,
    }


class SignedPermitServer:
    """Test-only wire peer that models the accepted HCP PermitServer calls."""

    def __init__(self, path: Path, *, count: int = 1, mode: str = "allow") -> None:
        self.path = path
        self.count = count
        self.mode = mode
        self.requests: list[dict[str, object]] = []
        self.error: BaseException | None = None
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(path))
        os.chmod(path, 0o600)
        self.listener.listen(count)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.thread.join(timeout=3)
        self.listener.close()
        self.path.unlink(missing_ok=True)
        assert not self.thread.is_alive()
        if self.error is not None:
            raise self.error

    def _serve(self) -> None:
        try:
            for index in range(self.count):
                channel, _ = self.listener.accept()
                with channel:
                    request = _read_frame(channel)
                    self.requests.append(request)
                    _write_frame(channel, self._response(request, index))
        except BaseException as exc:
            self.error = exc

    def _response(self, wire: dict[str, object], index: int) -> dict[str, object]:
        if wire.get("operation") == "bind_provider_response":
            return self._binding_response(wire, index)
        assert set(wire) == {
            "schema_version",
            "operation",
            "subject",
            "subject_sha256",
            "permit_exchange",
            "verify_exchange",
            "attestation_signature",
        }
        unsigned_wire = dict(wire)
        attestation = unsigned_wire.pop("attestation_signature")
        PEER_KEY.public_key().verify(
            bytes.fromhex(attestation), _canonical(unsigned_wire)
        )
        assert wire["subject_sha256"] == _digest(wire["subject"])

        exchanges = [wire["permit_exchange"], wire["verify_exchange"]]
        for exchange, operation in zip(exchanges, ("permit", "verify"), strict=True):
            assert exchange["operation"] == operation
            assert exchange["schema_version"] == "hcp.hermes.permit-exchange.v1"
            auth = {
                "schema_version": exchange["schema_version"],
                "operation": operation,
                "sender_identity": exchange["sender_identity"],
                "receiver_identity": exchange["receiver_identity"],
                "key_id": exchange["key_id"],
                "nonce": exchange["nonce"],
                "sequence": exchange["sequence"],
                "body": exchange["request"],
            }
            PEER_KEY.public_key().verify(
                bytes.fromhex(exchange["signature"]), _canonical(auth)
            )
        permit_request = exchanges[0]["request"]
        assert exchanges[1]["request"] == permit_request
        permit_unsigned = {
            "schema_version": "hcp.pre-model-permit.v2",
            "permit_id": (
                "permit-replayed"
                if self.mode == "permit_replay"
                else f"permit-{index + 1}"
            ),
            "request_digest": _digest(permit_request),
            "snapshot": permit_request["snapshot"],
            "auth_mode": permit_request["auth_mode"],
            "budget_enforcement_mode": permit_request["budget_enforcement_mode"],
            "nonce": permit_request["nonce"],
            "model_token_limit": permit_request["model_tokens_requested"],
            "retry_budget_charge": 0,
            "issued_at": permit_request["issued_at"],
            "expires_at": permit_request["expires_at"],
            "canonical_model_request_digest": _digest(
                permit_request["canonical_model_request"]
            ),
            "signing_identity": "hcp-permit-server",
            "key_id": "hcp-server-key-1",
        }
        if self.mode == "expired":
            expired = datetime.now(timezone.utc) - timedelta(seconds=1)
            permit_unsigned["expires_at"] = expired.isoformat().replace("+00:00", "Z")
        permit = {**permit_unsigned, "signature": _sign(SERVER_KEY, permit_unsigned)}
        now = datetime.now(timezone.utc) - timedelta(milliseconds=1)
        response_nonce = (
            "f" * 64 if self.mode == "response_replay" else f"{1000 + index:064x}"
        )
        unsigned = {
            "schema_version": "hermes.hcp.pre-model-permit-result.v1",
            "operation": "permit_and_verify_result",
            "subject_sha256": wire["subject_sha256"],
            "permit_exchange_nonce": exchanges[0]["nonce"],
            "verify_exchange_nonce": exchanges[1]["nonce"],
            "outcome": "PERMITTED",
            "error_code": None,
            "permit": permit,
            "verification": {"valid": True, "error_code": None},
            "server_identity": "hcp-permit-server",
            "server_key_id": "hcp-server-key-1",
            "response_nonce": response_nonce,
            "issued_at": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        }
        if self.mode == "deny":
            unsigned.update(
                outcome="NEUTRAL_HOLD",
                error_code="AUTHORITY_UNAVAILABLE",
                permit=None,
                verification={"valid": False, "error_code": "AUTHORITY_UNAVAILABLE"},
            )
        if self.mode == "wrong_subject":
            unsigned["subject_sha256"] = "sha256:" + "0" * 64
        response = {**unsigned, "signature": _sign(SERVER_KEY, unsigned)}
        if self.mode == "forged":
            response["signature"] = "0" * 128
        return response

    def _binding_response(
        self, wire: dict[str, object], index: int
    ) -> dict[str, object]:
        assert set(wire) == {
            "schema_version",
            "operation",
            "task_id",
            "session_id",
            "turn_id",
            "api_request_id",
            "authorization_id",
            "request_sha256",
            "subject_sha256",
            "response_observed_at_unix_ms",
            "response_sha256",
            "tool_calls",
            "signing_identity",
            "key_id",
            "issued_at_unix_ms",
            "nonce",
            "signature",
        }
        unsigned_wire = dict(wire)
        signature = unsigned_wire.pop("signature")
        PEER_KEY.public_key().verify(
            bytes.fromhex(signature), _canonical(unsigned_wire)
        )
        mirrored = {
            key: wire[key]
            for key in (
                "task_id",
                "session_id",
                "turn_id",
                "api_request_id",
                "authorization_id",
                "request_sha256",
                "subject_sha256",
                "response_observed_at_unix_ms",
                "response_sha256",
            )
        }
        binding_sha256 = _digest(unsigned_wire)
        response_nonce = (
            "e" * 64
            if self.mode == "binding_replay"
            else f"{2000 + index:064x}"
        )
        unsigned = {
            "schema_version": "hermes.hcp.provider-response-binding-result.v1",
            "operation": "bind_provider_response_result",
            **mirrored,
            "binding_sha256": binding_sha256,
            "binding_receipt_id": "binding-receipt:1",
            "status": "RECORDED",
            "error_code": None,
            "issued_at_unix_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
            "response_nonce": response_nonce,
            "signing_identity": "hcp-permit-server",
            "key_id": "hcp-server-key-1",
        }
        if self.mode == "binding_refuse":
            unsigned["status"] = "REFUSED"
            unsigned["error_code"] = "PROVIDER_RESPONSE_BINDING_REJECTED"
        if self.mode == "binding_mismatch":
            unsigned["tool_call_id"] = "unexpected"
        response = {**unsigned, "signature": _sign(SERVER_KEY, unsigned)}
        if self.mode == "binding_forged":
            response["signature"] = "0" * 128
        return response


def _guard(path: Path):
    from plugins.hcp_post_claim_pre_model_permit import HCPPermitGuard
    from plugins.hcp_post_claim_pre_model_permit.channel import (
        UnixSocketPermitTransport,
    )

    counter = itertools.count(1)
    return HCPPermitGuard(
        transport=UnixSocketPermitTransport(str(path)),
        manifest=_manifest(),
        peer_private_key=PEER_KEY,
        server_public_key=SERVER_KEY.public_key(),
        hermes_run_id="42",
        nonce_factory=lambda: f"{next(counter):064x}",
    )


def _request() -> dict[str, object]:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "exact\nrequest"}],
        "max_tokens": 31,
        "temperature": 0.25,
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_permit_channel_rejects_non_finite_json_numbers(value: float) -> None:
    from hermes_cli.provider_request_guard import ProviderRequestBlocked
    from plugins.hcp_post_claim_pre_model_permit.channel import canonical_json

    with pytest.raises(ProviderRequestBlocked, match="HCP_PERMIT_FRAME_INVALID"):
        canonical_json({"temperature": value})


@pytest.mark.parametrize("as_key", [False, True])
def test_permit_channel_rejects_non_utf8_string(as_key: bool) -> None:
    from hermes_cli.provider_request_guard import ProviderRequestBlocked
    from plugins.hcp_post_claim_pre_model_permit.channel import canonical_json

    with pytest.raises(ProviderRequestBlocked, match="HCP_PERMIT_FRAME_INVALID"):
        canonical_json({"\ud800": "prompt"} if as_key else {"prompt": "\ud800"})


def test_v2_codex_responses_manifest_is_accepted() -> None:
    from plugins.hcp_post_claim_pre_model_permit import HCPPermitGuard

    manifest = _manifest()
    manifest.update(
        {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "reasoning_effort": "high",
        }
    )
    guard = HCPPermitGuard(
        transport=SimpleNamespace(exchange=lambda _request: None),
        manifest=manifest,
        peer_private_key=PEER_KEY,
        server_public_key=SERVER_KEY.public_key(),
        hermes_run_id="42",
    )

    assert guard._manifest["schema_version"] == (
        "hermes.hcp.pre-model-permit-client.v2"
    )
    assert guard._manifest["api_mode"] == "codex_responses"
    assert guard._manifest["reasoning_effort"] == "high"
    assert guard._manifest["transport_mode"] == "non_streaming"


def test_v2_xai_oauth_responses_manifest_is_accepted() -> None:
    from plugins.hcp_post_claim_pre_model_permit import HCPPermitGuard

    manifest = _manifest()
    manifest.update(
        {
            "provider": "xai-oauth",
            "model": "grok-4.6",
            "api_mode": "codex_responses",
            "endpoint_origin": "https://api.x.ai",
            "reasoning_effort": "xhigh",
        }
    )
    guard = HCPPermitGuard(
        transport=SimpleNamespace(exchange=lambda _request: None),
        manifest=manifest,
        peer_private_key=PEER_KEY,
        server_public_key=SERVER_KEY.public_key(),
        hermes_run_id="42",
    )

    assert guard._manifest["provider"] == "xai-oauth"
    assert guard._manifest["model"] == "grok-4.6"
    assert guard._manifest["api_mode"] == "codex_responses"
    assert guard._manifest["endpoint_origin"] == "https://api.x.ai"
    assert guard._manifest["reasoning_effort"] == "xhigh"


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": "hermes.hcp.pre-model-permit-client.v1"},
        {"api_mode": "responses"},
        {"transport_mode": "streaming"},
        {"api_mode": "chat_completions", "reasoning_effort": "high"},
        {
            "api_mode": "codex_responses",
            "provider": "openai-codex",
            "reasoning_effort": None,
        },
        {
            "api_mode": "codex_responses",
            "provider": "openai",
            "reasoning_effort": "high",
        },
        {
            "api_mode": "codex_responses",
            "provider": "xai",
            "reasoning_effort": "xhigh",
        },
        {
            "api_mode": "codex_responses",
            "provider": "openai-codex",
            "reasoning_effort": "high\nlow",
        },
        {
            "api_mode": "codex_responses",
            "provider": "openai-codex",
            "reasoning_effort": "x" * 4097,
        },
    ],
)
def test_v2_manifest_refuses_inconsistent_route(
    changes: dict[str, object],
) -> None:
    from hermes_cli.provider_request_guard import ProviderRequestBlocked
    from plugins.hcp_post_claim_pre_model_permit import HCPPermitGuard

    manifest = _manifest()
    manifest.update(changes)
    with pytest.raises(ProviderRequestBlocked, match="HCP_PERMIT_MANIFEST_INVALID"):
        HCPPermitGuard(
            transport=SimpleNamespace(exchange=lambda _request: None),
            manifest=manifest,
            peer_private_key=PEER_KEY,
            server_public_key=SERVER_KEY.public_key(),
            hermes_run_id="42",
        )


def test_exact_hcp_exchanges_are_signed_and_each_attempt_gets_a_new_connection(
    tmp_path,
) -> None:
    server = SignedPermitServer(_socket_path(tmp_path), count=2)
    guard = _guard(server.path)
    request = _request()
    first = guard(request=request, request_sha256=_digest(request), **_context())
    second = guard(request=request, request_sha256=_digest(request), **_context())
    server.close()

    assert first.authorization_id == "permit-1"
    assert second.authorization_id == "permit-2"
    assert len(server.requests) == 2
    assert [
        exchange["sequence"]
        for wire in server.requests
        for exchange in (wire["permit_exchange"], wire["verify_exchange"])
    ] == [1, 2, 3, 4]
    observed = server.requests[0]
    assert observed["permit_exchange"]["request"]["canonical_model_request"] == request
    assert observed["subject"] | {} == {
        "schema_version": "hermes.hcp.worker-subject.v1",
        "task_id": "card-7",
        "hermes_run_id": "42",
        "hcp_run_id": "hcp-run-11",
        "bead_id": "hcp-rcv-007c",
        "session_id": "session-9",
        "profile_id": "hcp-general-implementer",
        "generation_id": "generation-3",
        "board_id": "hcp-board",
        "binding_digest": "sha256:" + "3" * 64,
        "claim_lock_sha256": _digest_text("host:123"),
        "worker_pid": os.getpid(),
        "turn_id": "turn-1",
        "api_request_id": "turn-1:api:1",
        "provider": "openai",
        "model": "test-model",
        "api_mode": "chat_completions",
        "endpoint_origin": "https://example.invalid",
        "transport_identity_sha256": "sha256:" + "d" * 64,
        "transport_mode": "non_streaming",
        "api_call_count": 1,
    }
    assert "test-only-key" not in json.dumps(observed)


def _response_binding(authorization) -> dict[str, object]:
    return {
        "task_id": "card-7",
        "session_id": "session-9",
        "turn_id": "turn-1",
        "api_request_id": "turn-1:api:1",
        "authorization_id": authorization.authorization_id,
        "request_sha256": authorization.request_sha256,
        "subject_sha256": authorization.subject_sha256,
        "response_observed_at_unix_ms": (
            int(datetime.now(timezone.utc).timestamp() * 1000) - 1
        ),
        "response_sha256": "sha256:" + "9" * 64,
        "tool_calls": [
            {
                "tool_call_id": "call:1",
                "tool_name": "hcp_diagnostics_read",
                "arguments_sha256": (
                    "sha256:a36609ff6956e75be3de9368842ae11618ac1dccecd30c337bda02665abae1d0"
                ),
            }
        ],
    }


def test_provider_response_binding_uses_separate_signed_closed_wire(tmp_path) -> None:
    server = SignedPermitServer(_socket_path(tmp_path), count=2)
    guard = _guard(server.path)
    request = _request()
    authorization = guard(
        request=request, request_sha256=_digest(request), **_context()
    )
    receipt = guard.bind_provider_response(**_response_binding(authorization))
    server.close()

    assert receipt["binding_sha256"].startswith("sha256:")
    assert receipt["binding_receipt_id"] == "binding-receipt:1"
    wire = server.requests[1]
    assert wire["schema_version"] == "hermes.hcp.provider-response-binding.v1"
    assert wire["operation"] == "bind_provider_response"
    assert wire["tool_calls"][0] == {
        "tool_call_id": "call:1",
        "tool_name": "hcp_diagnostics_read",
        "arguments_sha256": (
            "sha256:a36609ff6956e75be3de9368842ae11618ac1dccecd30c337bda02665abae1d0"
        ),
    }
    assert set(wire) == {
        "schema_version", "operation", "task_id", "session_id", "turn_id",
        "api_request_id", "authorization_id", "request_sha256",
        "subject_sha256", "response_observed_at_unix_ms", "response_sha256",
        "tool_calls", "signing_identity", "key_id", "issued_at_unix_ms",
        "nonce", "signature",
    }


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("binding_refuse", "PROVIDER_RESPONSE_BINDING_REJECTED"),
        ("binding_mismatch", "PROVIDER_RESPONSE_BINDING_INVALID"),
        ("binding_forged", "HCP_PERMIT_SIGNATURE_INVALID"),
    ],
)
def test_provider_response_binding_refusal_mismatch_and_forgery_fail_closed(
    tmp_path, mode: str, error: str
) -> None:
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    server = SignedPermitServer(_socket_path(tmp_path), count=2, mode=mode)
    guard = _guard(server.path)
    request = _request()
    authorization = guard(
        request=request, request_sha256=_digest(request), **_context()
    )
    with pytest.raises(ProviderRequestBlocked, match=error):
        guard.bind_provider_response(**_response_binding(authorization))
    server.close()


def test_exact_idempotent_binding_is_allowed_but_response_replay_is_not(
    tmp_path,
) -> None:
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    server = SignedPermitServer(_socket_path(tmp_path), count=3, mode="binding_replay")
    guard = _guard(server.path)
    request = _request()
    authorization = guard(
        request=request, request_sha256=_digest(request), **_context()
    )
    binding = _response_binding(authorization)
    guard.bind_provider_response(**binding)
    with pytest.raises(ProviderRequestBlocked, match="REPLAY"):
        guard.bind_provider_response(**binding)
    server.close()


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("deny", "AUTHORITY_UNAVAILABLE"),
        ("expired", "HCP_PERMIT_RESPONSE_MISMATCH"),
        ("wrong_subject", "HCP_PERMIT_RESPONSE_INVALID"),
        ("forged", "HCP_PERMIT_SIGNATURE_INVALID"),
    ],
)
def test_denial_expiry_identity_and_signature_fail_closed(
    tmp_path, mode: str, error: str
) -> None:
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    server = SignedPermitServer(_socket_path(tmp_path), mode=mode)
    guard = _guard(server.path)
    request = _request()
    with pytest.raises(ProviderRequestBlocked, match=error):
        guard(request=request, request_sha256=_digest(request), **_context())
    server.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", "card-8"),
        ("profile_id", "other-profile"),
        ("provider", "other-provider"),
        ("model", "other-model"),
        ("endpoint_origin", "https://other.invalid"),
        ("transport_mode", "streaming"),
        ("api_call_count", 0),
    ],
)
def test_caller_identity_mismatch_never_contacts_hcp(
    field: str, value: object
) -> None:
    from hermes_cli.provider_request_guard import ProviderRequestBlocked
    from plugins.hcp_post_claim_pre_model_permit import HCPPermitGuard

    transport = SimpleNamespace(exchange=lambda _request: pytest.fail("contacted HCP"))
    counter = itertools.count(1)
    guard = HCPPermitGuard(
        transport=transport,
        manifest=_manifest(),
        peer_private_key=PEER_KEY,
        server_public_key=SERVER_KEY.public_key(),
        hermes_run_id="42",
        nonce_factory=lambda: f"{next(counter):064x}",
    )
    context = _context()
    context[field] = value
    request = _request()
    if field == "model":
        request["model"] = value
    with pytest.raises(ProviderRequestBlocked, match="IDENTITY_MISMATCH"):
        guard(request=request, request_sha256=_digest(request), **context)


def test_session_rebind_after_first_permit_is_denied_without_a_second_contact(
    tmp_path,
) -> None:
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    server = SignedPermitServer(_socket_path(tmp_path))
    guard = _guard(server.path)
    request = _request()
    guard(request=request, request_sha256=_digest(request), **_context())
    server.close()
    rebound = _context()
    rebound["session_id"] = "session-other"
    with pytest.raises(ProviderRequestBlocked, match="IDENTITY_MISMATCH"):
        guard(request=request, request_sha256=_digest(request), **rebound)


@pytest.mark.parametrize("mode", ["response_replay", "permit_replay"])
def test_server_response_and_permit_replay_are_denied(tmp_path, mode: str) -> None:
    from hermes_cli.provider_request_guard import ProviderRequestBlocked

    server = SignedPermitServer(_socket_path(tmp_path), count=2, mode=mode)
    guard = _guard(server.path)
    request = _request()
    guard(request=request, request_sha256=_digest(request), **_context())
    with pytest.raises(ProviderRequestBlocked, match="REPLAY"):
        guard(request=request, request_sha256=_digest(request), **_context())
    server.close()


def test_plugin_reads_and_closes_only_protected_worker_inputs(
    tmp_path, monkeypatch
) -> None:
    from plugins.hcp_post_claim_pre_model_permit import register

    server = SignedPermitServer(_socket_path(tmp_path))
    manifest_path = tmp_path / "manifest.json"
    private_path = tmp_path / "peer.key"
    public_path = tmp_path / "server.pub"
    manifest_path.write_bytes(_canonical(_manifest()))
    private_path.write_bytes(b"p" * 32)
    public_path.write_bytes(SERVER_KEY.public_key().public_bytes_raw())
    for path in (manifest_path, private_path, public_path):
        path.chmod(0o400)
    fds = [
        os.open(path, os.O_RDONLY)
        for path in (manifest_path, private_path, public_path)
    ]
    monkeypatch.setenv("HCP_PRE_MODEL_PERMIT_SOCKET", str(server.path))
    monkeypatch.setenv("HCP_PRE_MODEL_PERMIT_MANIFEST_FD", str(fds[0]))
    monkeypatch.setenv("HCP_PRE_MODEL_PERMIT_PEER_PRIVATE_KEY_FD", str(fds[1]))
    monkeypatch.setenv("HCP_PRE_MODEL_PERMIT_SERVER_PUBLIC_KEY_FD", str(fds[2]))
    monkeypatch.setenv("HCP_PRE_MODEL_PERMIT_REQUIRED", "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "card-7")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:123")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "hcp-board")
    monkeypatch.setenv("HERMES_PROFILE", "hcp-general-implementer")
    captured = {}
    ctx = SimpleNamespace(
        profile_name="hcp-general-implementer",
        register_provider_request_guard=lambda callback: captured.update(
            guard=callback
        ),
    )

    register(ctx)
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    for name in (
        "HCP_PRE_MODEL_PERMIT_SOCKET",
        "HCP_PRE_MODEL_PERMIT_MANIFEST_FD",
        "HCP_PRE_MODEL_PERMIT_PEER_PRIVATE_KEY_FD",
        "HCP_PRE_MODEL_PERMIT_SERVER_PUBLIC_KEY_FD",
        "HCP_PRE_MODEL_PERMIT_REQUIRED",
    ):
        assert name not in os.environ
    request = _request()
    captured["guard"](
        request=request,
        request_sha256=_digest(request),
        **_context(),
    )
    server.close()


def test_malformed_manifest_still_closes_every_capability_fd(
    tmp_path, monkeypatch
) -> None:
    from plugins.hcp_post_claim_pre_model_permit import register

    paths = [tmp_path / "manifest", tmp_path / "peer", tmp_path / "server"]
    paths[0].write_bytes(b"{}")
    paths[1].write_bytes(b"p" * 32)
    paths[2].write_bytes(b"s" * 32)
    for path in paths:
        path.chmod(0o400)
    fds = [os.open(path, os.O_RDONLY) for path in paths]
    monkeypatch.setenv("HCP_PRE_MODEL_PERMIT_SOCKET", "/tmp/not-contacted.sock")
    monkeypatch.setenv("HCP_PRE_MODEL_PERMIT_MANIFEST_FD", str(fds[0]))
    monkeypatch.setenv("HCP_PRE_MODEL_PERMIT_PEER_PRIVATE_KEY_FD", str(fds[1]))
    monkeypatch.setenv("HCP_PRE_MODEL_PERMIT_SERVER_PUBLIC_KEY_FD", str(fds[2]))
    ctx = SimpleNamespace(
        profile_name="hcp-general-implementer",
        register_provider_request_guard=lambda _callback: None,
    )

    with pytest.raises(RuntimeError, match="MANIFEST_INVALID"):
        register(ctx)
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_required_guard_has_one_exact_plugin_owner() -> None:
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from hermes_cli.provider_request_guard import ProviderRequestGuardRegistrationError

    manager = PluginManager()
    manager._provider_request_guard_required = True
    with pytest.raises(ProviderRequestGuardRegistrationError, match="wrong owner"):
        manager.register_provider_request_guard(lambda **_: None, owner="other-plugin")
    with pytest.raises(ProviderRequestGuardRegistrationError, match="capability"):
        manager.register_provider_request_guard(
            lambda **_: None, owner="hcp_post_claim_pre_model_permit"
        )
    attacker = PluginContext(
        PluginManifest(
            name="hcp_post_claim_pre_model_permit",
            key="hcp_post_claim_pre_model_permit",
            source="user",
            path="/tmp/untrusted-hcp-plugin",
        ),
        manager,
    )
    with pytest.raises(ProviderRequestGuardRegistrationError, match="capability"):
        attacker.register_provider_request_guard(lambda **_: None)


def test_required_mode_rejects_a_same_name_non_bundled_manifest() -> None:
    from hermes_cli.plugins import PluginManager, PluginManifest
    from hermes_cli.provider_request_guard import ProviderRequestGuardRegistrationError

    manager = PluginManager()
    manager._provider_request_guard_required = True
    attacker = PluginManifest(
        name="hcp_post_claim_pre_model_permit",
        key="hcp_post_claim_pre_model_permit",
        source="entrypoint",
        path="attacker.module:register",
    )

    with pytest.raises(
        ProviderRequestGuardRegistrationError, match="reserved plugin identity"
    ):
        manager._validate_required_provider_guard_manifests([attacker])


def test_required_manifest_is_exact_package_owned_content(
    tmp_path, monkeypatch
) -> None:
    from hermes_cli.plugins import PluginManager

    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "redirected"))
    manager = PluginManager()
    manager._provider_request_guard_required = True

    manifest = manager._required_provider_guard_manifest()

    assert manifest.source == "bundled"
    assert manifest.name == "hcp_post_claim_pre_model_permit"
    assert Path(manifest.path).resolve() != (tmp_path / "redirected").resolve()
    assert manager._is_exact_required_provider_guard(manifest) is True


def test_required_mode_imports_only_the_exact_package_guard(monkeypatch) -> None:
    from hermes_cli.plugins import PluginManager, PluginManifest

    manager = PluginManager()
    manager._provider_request_guard_required = True
    unrelated = PluginManifest(
        name="untrusted-observer",
        key="untrusted-observer",
        source="entrypoint",
        path="attacker.module:register",
    )
    exact = manager._required_provider_guard_manifest()
    loaded: list[str] = []

    monkeypatch.setattr(manager, "_scan_directory", lambda *args, **kwargs: [])
    monkeypatch.setattr(manager, "_scan_entry_points", lambda: [unrelated])
    monkeypatch.setattr(manager, "_required_provider_guard_manifest", lambda: exact)

    def load_plugin(manifest) -> None:
        loaded.append(manifest.name)
        manager.register_provider_request_guard(
            lambda **_: None,
            owner=manifest.name,
            capability=manager._PluginManager__provider_guard_registration_capability,
        )

    monkeypatch.setattr(manager, "_load_plugin", load_plugin)

    manager._discover_and_load_inner()

    assert loaded == ["hcp_post_claim_pre_model_permit"]


def _task(tmp_path: Path):
    from hermes_cli.kanban_db import Task

    return Task(
        id="card-7",
        title="permit boundary",
        body=None,
        assignee="hcp-general-implementer",
        status="running",
        priority=0,
        created_by="hcp",
        created_at=0,
        started_at=0,
        completed_at=None,
        workspace_kind="existing",
        workspace_path=str(tmp_path),
        claim_lock="host:123",
        claim_expires=100,
        tenant=None,
        current_run_id=42,
    )


def test_dispatcher_selects_task_files_and_never_shares_its_root_fd(
    tmp_path, monkeypatch
) -> None:
    from hermes_cli import kanban_db

    capability_root = tmp_path / "capabilities"
    capability_root.mkdir(mode=0o700)
    stem = kanban_db._hcp_worker_input_stem("card-7")
    assert stem != kanban_db._hcp_worker_input_stem("card-8")
    files = {
        f"{stem}.manifest.json": b"manifest",
        f"{stem}.peer.ed25519": b"p" * 32,
        "server.ed25519.pub": b"s" * 32,
    }
    for name, content in files.items():
        path = capability_root / name
        path.write_bytes(content)
        path.chmod(0o400)
    root_fd = os.open(capability_root, os.O_RDONLY)
    monkeypatch.setenv("HCP_PRE_MODEL_PERMIT_ROOT_FD", str(root_fd))
    monkeypatch.setenv("HCP_PRE_MODEL_PERMIT_SOCKET", str(tmp_path / "permit.sock"))
    captured = {}

    class FakeProcess:
        pid = 12345

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        captured["contents"] = [os.pread(fd, 100, 0) for fd in kwargs["pass_fds"]]
        return FakeProcess()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(kanban_db, "worker_logs_dir", lambda **_: tmp_path)
    monkeypatch.setattr(kanban_db, "_resolve_worker_cli_toolsets", lambda _home: None)
    kanban_db._default_spawn(_task(tmp_path), str(tmp_path), board="hcp-board")

    assert captured["contents"] == [b"manifest", b"p" * 32, b"s" * 32]
    assert root_fd not in captured["pass_fds"]
    assert "HCP_PRE_MODEL_PERMIT_ROOT_FD" not in captured["env"]
    assert captured["env"]["HCP_PRE_MODEL_PERMIT_REQUIRED"] == "1"
    assert captured["env"]["HERMES_KANBAN_TASK"] == "card-7"
    assert captured["env"]["HERMES_KANBAN_RUN_ID"] == "42"
    for fd in captured["pass_fds"]:
        with pytest.raises(OSError):
            os.fstat(fd)
    os.close(root_fd)


def test_missing_exact_task_capability_refuses_before_spawn(
    tmp_path, monkeypatch
) -> None:
    from hermes_cli import kanban_db

    capability_root = tmp_path / "capabilities"
    capability_root.mkdir(mode=0o700)
    root_fd = os.open(capability_root, os.O_RDONLY)
    monkeypatch.setenv("HCP_PRE_MODEL_PERMIT_ROOT_FD", str(root_fd))
    monkeypatch.setenv("HCP_PRE_MODEL_PERMIT_SOCKET", str(tmp_path / "permit.sock"))
    popen = SimpleNamespace(called=False)
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda *_args, **_kwargs: setattr(popen, "called", True),
    )
    monkeypatch.setattr(kanban_db, "worker_logs_dir", lambda **_: tmp_path)
    monkeypatch.setattr(kanban_db, "_resolve_worker_cli_toolsets", lambda _home: None)
    with pytest.raises(RuntimeError, match="WORKER_INPUT_INVALID"):
        kanban_db._default_spawn(_task(tmp_path), str(tmp_path), board="hcp-board")
    assert popen.called is False
    os.close(root_fd)


def test_profile_dotenv_cannot_clear_or_replace_managed_worker_identity(
    tmp_path, monkeypatch
) -> None:
    from hermes_cli.env_loader import load_hermes_dotenv

    original = {
        "HCP_PRE_MODEL_PERMIT_REQUIRED": "1",
        "HCP_PRE_MODEL_PERMIT_SOCKET": "/private/permit.sock",
        "HCP_PRE_MODEL_PERMIT_MANIFEST_FD": "71",
        "HCP_PRE_MODEL_PERMIT_PEER_PRIVATE_KEY_FD": "72",
        "HCP_PRE_MODEL_PERMIT_SERVER_PUBLIC_KEY_FD": "73",
        "HERMES_KANBAN_TASK": "card-7",
        "HERMES_KANBAN_RUN_ID": "42",
        "HERMES_KANBAN_CLAIM_LOCK": "host:123",
        "HERMES_KANBAN_BOARD": "hcp-board",
        "HERMES_KANBAN_DB": "/private/board.db",
        "HERMES_PROFILE": "hcp-general-implementer",
        "TERMINAL_DOCKER_REQUIRE_RESOURCE_LIMITS": "1",
    }
    for key, value in original.items():
        monkeypatch.setenv(key, value)
    (tmp_path / ".env").write_text(
        "\n".join(f"{key}=attacker-value" for key in original)
        + "\nHCP_PRE_MODEL_PERMIT_ROOT_FD=74\n",
        encoding="utf-8",
    )

    load_hermes_dotenv(hermes_home=tmp_path)

    assert {key: os.environ[key] for key in original} == original
    assert "HCP_PRE_MODEL_PERMIT_ROOT_FD" not in os.environ


def test_pre_spawn_capability_refusal_is_a_zero_charge_claim_hold(
    tmp_path,
) -> None:
    from hermes_cli import kanban_db

    db_path = tmp_path / "kanban.db"
    with kanban_db.connect(db_path) as conn:
        task_id = kanban_db.create_task(
            conn,
            title="guarded spawn",
            assignee="hcp-general-implementer",
            initial_status="blocked",
        )
        assert kanban_db.promote_task(conn, task_id, actor="test", force=True)[0]
        task = kanban_db.claim_task(conn, task_id, claimer="host:123")
        assert task is not None and task.current_run_id is not None

        result = kanban_db.record_claimed_provider_guard_hold(
            conn,
            task,
            "HCP_PERMIT_WORKER_INPUT_INVALID",
        )

        assert result["failed"] is False
        assert result["integrity_hold"] is True
        held = kanban_db.get_task(conn, task_id)
        assert held.status == "blocked"
        assert held.block_kind == "provider_guard_integrity"
        assert held.consecutive_failures == 0
        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE id = ?",
            (task.current_run_id,),
        ).fetchone()
        assert run["outcome"] == "blocked"
        assert json.loads(run["metadata"])["provider_guard_hold_kind"] == ("integrity")


def test_dispatcher_does_not_charge_a_capability_input_refusal(
    tmp_path, monkeypatch
) -> None:
    from hermes_cli import kanban_db, profiles

    db_path = tmp_path / "kanban.db"
    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    monkeypatch.setattr(
        kanban_db, "resolve_workspace", lambda _task, **_kwargs: tmp_path
    )

    def refuse(_task, _workspace, **_kwargs):
        raise kanban_db.ProviderGuardSpawnBlocked("HCP_PERMIT_WORKER_INPUT_INVALID")

    with kanban_db.connect(db_path) as conn:
        task_id = kanban_db.create_task(
            conn,
            title="guarded dispatch",
            assignee="hcp-general-implementer",
            initial_status="blocked",
        )
        assert kanban_db.promote_task(conn, task_id, actor="test", force=True)[0]

        result = kanban_db._dispatch_once_locked(conn, spawn_fn=refuse)

        assert result.spawned == []
        assert result.auto_blocked == []
        held = kanban_db.get_task(conn, task_id)
        assert held.status == "blocked"
        assert held.block_kind == "provider_guard_integrity"
        assert held.consecutive_failures == 0


def test_permit_refusal_records_one_neutral_hold_without_failure_charge(
    tmp_path, monkeypatch
) -> None:
    from hermes_cli import kanban_db

    db_path = tmp_path / "kanban.db"
    with kanban_db.connect(db_path) as conn:
        task_id = kanban_db.create_task(
            conn,
            title="guarded",
            assignee="hcp-general-implementer",
            initial_status="blocked",
        )
        assert kanban_db.promote_task(conn, task_id, actor="test", force=True)[0]
        task = kanban_db.claim_task(conn, task_id, claimer="host:123")
        assert task is not None and task.current_run_id is not None
        run_id = task.current_run_id
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "hcp-board")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:123")
    monkeypatch.setenv("HERMES_PROFILE", "hcp-general-implementer")

    result = kanban_db.record_provider_guard_hold("AUTHORITY_UNAVAILABLE")
    assert result == {
        "final_response": "",
        "messages": [],
        "api_calls": 0,
        "completed": False,
        "failed": False,
        "neutral_hold": True,
        "failure_reason": "neutral_hold",
        "error": "AUTHORITY_UNAVAILABLE",
    }
    assert kanban_db.record_provider_guard_hold("AUTHORITY_UNAVAILABLE") is None
    with kanban_db.connect(db_path) as conn:
        task = kanban_db.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.consecutive_failures == 0
        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert run["outcome"] == "blocked"
        assert json.loads(run["metadata"])["provider_guard_hold"] is True
        events = conn.execute(
            "SELECT count(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (task_id, "provider_guard_hold"),
        ).fetchone()[0]
        assert events == 1


def test_neutral_hold_never_creates_a_missing_tracker(tmp_path, monkeypatch) -> None:
    from hermes_cli import kanban_db

    missing = tmp_path / "missing.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(missing))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "hcp-board")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "card-7")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:123")
    monkeypatch.setenv("HERMES_PROFILE", "hcp-general-implementer")
    assert kanban_db.record_provider_guard_hold("AUTHORITY_UNAVAILABLE") is None
    assert not missing.exists()


def test_signature_or_replay_refusal_is_a_sticky_integrity_hold(
    tmp_path, monkeypatch
) -> None:
    from hermes_cli import kanban_db

    db_path = tmp_path / "kanban.db"
    with kanban_db.connect(db_path) as conn:
        task_id = kanban_db.create_task(
            conn,
            title="integrity",
            assignee="hcp-general-implementer",
            initial_status="blocked",
        )
        assert kanban_db.promote_task(conn, task_id, actor="test", force=True)[0]
        task = kanban_db.claim_task(conn, task_id, claimer="host:123")
        assert task is not None and task.current_run_id is not None
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "hcp-board")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:123")
    monkeypatch.setenv("HERMES_PROFILE", "hcp-general-implementer")

    result = kanban_db.record_provider_guard_hold("HCP_PERMIT_SIGNATURE_INVALID")

    assert result["failed"] is False
    assert result["neutral_hold"] is False
    assert result["integrity_hold"] is True
    with kanban_db.connect(db_path) as conn:
        held = kanban_db.get_task(conn, task_id)
        assert held.block_kind == "provider_guard_integrity"
        assert held.consecutive_failures == 0
        assert kanban_db.recompute_ready(conn) == 0
        assert kanban_db.get_task(conn, task_id).status == "blocked"
