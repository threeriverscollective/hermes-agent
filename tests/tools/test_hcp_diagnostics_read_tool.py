from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import tempfile
import threading
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from hermes_cli.provider_request_guard import (
    ProviderRequestAuthorization,
    ProviderRequestBlocked,
    begin_provider_request_authorization,
    bind_provider_response_tool_calls,
    canonical_request_sha256,
    consume_provider_tool_invocation,
    discard_provider_request_authorization,
)
from model_tools import _clear_tool_defs_cache, get_tool_definitions
from tools.hcp_diagnostics_read_tool import (
    CONFIG_SCHEMA_VERSION,
    HCP_DIAGNOSTICS_READ_SCHEMA,
    RESULT_SCHEMA_VERSION,
    TOOL_NAME,
    WIRE_SCHEMA_VERSION,
    hcp_diagnostics_available,
    hcp_diagnostics_read_tool,
)
from tools.registry import registry
from toolsets import TOOLSETS, _HERMES_CORE_TOOLS


IDENTITY = {
    "task_id": "card:1",
    "session_id": "session:1",
    "turn_id": "turn:1",
    "api_request_id": "api:1",
    "tool_call_id": "call:1",
}


@pytest.fixture()
def tmp_path():
    """Use a compact owner-only root that fits Darwin's AF_UNIX limit."""

    path = Path(tempfile.mkdtemp(prefix="hcpdt-", dir="/private/tmp"))
    path.chmod(0o700)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _owner_only_ledger_home(tmp_path: Path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        "hermes_cli.plugins.bind_provider_response_guard",
        lambda **binding: {
            "binding_sha256": canonical_request_sha256(binding),
            "binding_receipt_id": "binding-receipt:test",
        },
    )
    yield


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _write_private(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    path.chmod(0o600)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _binding(tmp_path: Path):
    tmp_path.chmod(0o700)
    request_key = Ed25519PrivateKey.generate()
    response_key = Ed25519PrivateKey.generate()
    request_path = tmp_path / "request.key"
    response_path = tmp_path / "response.pub"
    request_sha = _write_private(request_path, request_key.private_bytes_raw())
    response_sha = _write_private(
        response_path, response_key.public_key().public_bytes_raw()
    )
    config_path = tmp_path / "tool-config.json"
    config = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "repository_id": "repository-a",
        "card_id": "card:1",
        "run_id": "run:1",
        "socket_path": str(tmp_path / "broker.sock"),
        "request_signing": {
            "signing_identity": "hermes:bridge:repository-a",
            "key_id": "key:bridge:1",
            "key_path": str(request_path),
            "key_sha256": request_sha,
        },
        "response_verification": {
            "signing_identity": "hcp:broker:repository-a",
            "key_id": "key:hcp:1",
            "key_path": str(response_path),
            "key_sha256": response_sha,
        },
        "timeout_ms": 2_000,
    }
    _write_private(config_path, _canonical(config))
    return config_path, config, request_key, response_key


def _authorize(identity: dict[str, str] = IDENTITY) -> None:
    authorization = ProviderRequestAuthorization(
        authorization_id="permit:1",
        request_sha256="sha256:" + "a" * 64,
        subject_sha256="sha256:" + "b" * 64,
        expires_at_monotonic=time.monotonic() + 10,
    )
    request_identity = {
        key: identity[key]
        for key in ("task_id", "session_id", "turn_id", "api_request_id")
    }
    begin_provider_request_authorization(
        **request_identity,
        authorization=authorization,
    )
    bind_provider_response_tool_calls(
        **request_identity,
        tool_calls=[
            (identity["tool_call_id"], TOOL_NAME, '{"offer_id":"diagnostic-offer:1111111111111111111111111111111111111111111111111111111111111111"}')
        ],
        finish_reason="tool_calls",
        assistant_content="",
        response_observed_at_unix_ms=1_786_000_000_000,
    )


def _discard(identity: dict[str, str] = IDENTITY) -> None:
    discard_provider_request_authorization(
        **{
            key: identity[key]
            for key in ("task_id", "session_id", "turn_id", "api_request_id")
        }
    )


def _serve_once(
    socket_path: Path,
    *,
    request_key: Ed25519PrivateKey,
    response_key: Ed25519PrivateKey,
    invalid_signature: bool = False,
):
    ready = threading.Event()
    observed: list[dict[str, object]] = []

    def target() -> None:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(socket_path))
            os.chmod(socket_path, 0o600)
            listener.listen(1)
            ready.set()
            channel, _ = listener.accept()
            with channel:
                size = struct.unpack(">I", channel.recv(4))[0]
                raw = b""
                while len(raw) < size:
                    raw += channel.recv(size - len(raw))
                request = json.loads(raw)
                observed.append(request)
                unsigned_request = dict(request)
                signature = unsigned_request.pop("signature")
                request_key.public_key().verify(
                    bytes.fromhex(signature), _canonical(unsigned_request)
                )
                echoed = {
                    key: request[key]
                    for key in (
                        "repository_id",
                        "card_id",
                        "run_id",
                        "offer_id",
                        "task_id",
                        "session_id",
                        "turn_id",
                        "api_request_id",
                        "tool_call_id",
                        "provider_authorization_id",
                        "provider_request_sha256",
                        "provider_subject_sha256",
                        "provider_response_observed_at_unix_ms",
                        "provider_response_sha256",
                        "provider_response_binding_sha256",
                        "provider_response_binding_receipt_id",
                    )
                }
                unsigned_response = {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "status": "ok",
                    "error_code": None,
                    **echoed,
                    "records": [{"number": 42, "title": "bounded result"}],
                    "receipt": {"receipt_sha256": "sha256:" + "d" * 64},
                    "replayed": False,
                    "signing_identity": "hcp:broker:repository-a",
                    "key_id": "key:hcp:1",
                }
                signer = (
                    Ed25519PrivateKey.generate()
                    if invalid_signature
                    else response_key
                )
                response = {
                    **unsigned_response,
                    "signature": signer.sign(_canonical(unsigned_response)).hex(),
                }
                payload = _canonical(response)
                channel.sendall(struct.pack(">I", len(payload)) + payload)
        finally:
            listener.close()
            if socket_path.exists():
                socket_path.unlink()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    assert ready.wait(timeout=2)
    return thread, observed


def test_tool_is_unavailable_without_exact_hcp_runtime_binding(monkeypatch) -> None:
    monkeypatch.delenv("HCP_DIAGNOSTICS_TOOL_CONFIG", raising=False)
    assert hcp_diagnostics_available() is False


def test_schema_exposes_only_stable_opaque_offer_id() -> None:
    parameters = HCP_DIAGNOSTICS_READ_SCHEMA["parameters"]
    assert set(parameters["properties"]) == {"offer_id"}
    assert parameters["required"] == ["offer_id"]
    assert parameters["additionalProperties"] is False


def test_cross_repository_offer_argument_digest_vector_is_exact(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def bind(**binding):
        captured.append(binding)
        return {
            "binding_sha256": canonical_request_sha256(binding),
            "binding_receipt_id": "binding-receipt:vector",
        }

    monkeypatch.setattr("hermes_cli.plugins.bind_provider_response_guard", bind)
    _authorize()
    assert captured[0]["tool_calls"] == [
        {
            "tool_call_id": "call:1",
            "tool_name": "hcp_diagnostics_read",
            "arguments_sha256": (
                "sha256:2e2201f66544a45d32ae9866e6e02bd86a8bc9f3fed70ec69e90dca5a401956c"
            ),
        }
    ]
    _discard()


def test_tool_inventory_is_one_pinned_opt_in_toolset() -> None:
    assert TOOLSETS["hcp_diagnostics"] == {
        "description": (
            "HCP-admitted repository-scoped diagnostic reads over a private "
            "authenticated host bridge"
        ),
        "tools": ["hcp_diagnostics_read"],
        "includes": [],
    }
    assert "hcp_diagnostics_read" not in _HERMES_CORE_TOOLS
    entry = registry.get_entry("hcp_diagnostics_read")
    assert entry is not None
    assert entry.toolset == "hcp_diagnostics"
    assert entry.schema == HCP_DIAGNOSTICS_READ_SCHEMA


def test_first_pilot_inventory_is_exact_terminal_plus_host_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, _, _, _ = _binding(tmp_path)
    monkeypatch.setenv("HCP_DIAGNOSTICS_TOOL_CONFIG", str(config_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    _clear_tool_defs_cache()
    try:
        exposed = get_tool_definitions(
            enabled_toolsets=["terminal", "hcp_diagnostics"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        without_diagnostics = get_tool_definitions(
            enabled_toolsets=["terminal"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
    finally:
        _clear_tool_defs_cache()
    assert sorted(row["function"]["name"] for row in exposed) == [
        "hcp_diagnostics_read",
        "process",
        "terminal",
    ]
    assert sorted(row["function"]["name"] for row in without_diagnostics) == [
        "process",
        "terminal",
    ]


def test_tool_crosses_socket_with_host_identity_and_provider_attestation(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, config, request_key, response_key = _binding(tmp_path)
    monkeypatch.setenv("HCP_DIAGNOSTICS_TOOL_CONFIG", str(config_path))
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            config_path,
            Path(config["request_signing"]["key_path"]),
            Path(config["response_verification"]["key_path"]),
        )
    }
    _authorize()
    thread, observed = _serve_once(
        Path(config["socket_path"]),
        request_key=request_key,
        response_key=response_key,
    )

    result = json.loads(
        hcp_diagnostics_read_tool({"offer_id": "diagnostic-offer:1111111111111111111111111111111111111111111111111111111111111111"}, **IDENTITY)
    )
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert result["status"] == "ok"
    assert result["records"] == [{"number": 42, "title": "bounded result"}]
    assert len(observed) == 1
    request = observed[0]
    assert request["schema_version"] == WIRE_SCHEMA_VERSION
    assert request["repository_id"] == "repository-a"
    assert request["card_id"] == "card:1"
    assert request["run_id"] == "run:1"
    assert request["offer_id"] == "diagnostic-offer:1111111111111111111111111111111111111111111111111111111111111111"
    assert {key: request[key] for key in IDENTITY} == IDENTITY
    assert request["provider_authorization_id"] == "permit:1"
    assert request["provider_request_sha256"] == "sha256:" + "a" * 64
    assert request["provider_subject_sha256"] == "sha256:" + "b" * 64
    assert request["provider_response_observed_at_unix_ms"] == 1_786_000_000_000
    assert request["provider_response_sha256"].startswith("sha256:")
    assert request["provider_response_binding_sha256"].startswith("sha256:")
    assert request["provider_response_binding_receipt_id"] == "binding-receipt:test"
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before
    } == before


def test_model_authored_authority_is_rejected_before_socket_access(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, _, _, _ = _binding(tmp_path)
    monkeypatch.setenv("HCP_DIAGNOSTICS_TOOL_CONFIG", str(config_path))
    result = json.loads(
        hcp_diagnostics_read_tool(
            {"offer_id": "diagnostic-offer:1111111111111111111111111111111111111111111111111111111111111111", "repository_id": "repository-b"},
            **IDENTITY,
        )
    )
    assert result["error_code"] == "BROKER_SCOPE_MISMATCH"


def test_direct_tool_call_without_authorized_response_is_refused_before_socket(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, _, _, _ = _binding(tmp_path)
    monkeypatch.setenv("HCP_DIAGNOSTICS_TOOL_CONFIG", str(config_path))
    _discard()
    result = json.loads(
        hcp_diagnostics_read_tool({"offer_id": "diagnostic-offer:1111111111111111111111111111111111111111111111111111111111111111"}, **IDENTITY)
    )
    assert result["error_code"] == "BROKER_TRANSPORT_AUTH_INVALID"


def test_provider_tool_attestation_is_exact_and_one_shot() -> None:
    _authorize()
    wrong = {**IDENTITY, "tool_call_id": "call:other"}
    try:
        try:
            consume_provider_tool_invocation(**wrong, tool_name=TOOL_NAME)
        except ProviderRequestBlocked as error:
            assert error.error_code == "PROVIDER_TOOL_INVOCATION_UNATTESTED"
        else:
            raise AssertionError("wrong tool-call identity was accepted")

        attestation = consume_provider_tool_invocation(
            **IDENTITY, tool_name=TOOL_NAME
        )
        assert attestation.authorization_id == "permit:1"
        assert attestation.response_binding_sha256.startswith("sha256:")
        try:
            consume_provider_tool_invocation(**IDENTITY, tool_name=TOOL_NAME)
        except ProviderRequestBlocked as error:
            assert error.error_code == "PROVIDER_TOOL_INVOCATION_UNATTESTED"
        else:
            raise AssertionError("provider tool attestation replay was accepted")
    finally:
        _discard()


def test_slow_authentic_response_can_bind_after_initiation_permit_expires() -> None:
    request_identity = {
        key: IDENTITY[key]
        for key in ("task_id", "session_id", "turn_id", "api_request_id")
    }
    begin_provider_request_authorization(
        **request_identity,
        authorization=ProviderRequestAuthorization(
            authorization_id="permit:slow",
            request_sha256="sha256:" + "c" * 64,
            subject_sha256="sha256:" + "d" * 64,
            expires_at_monotonic=time.monotonic() + 0.01,
        ),
    )
    time.sleep(0.02)
    bind_provider_response_tool_calls(
        **request_identity,
        tool_calls=[
            (IDENTITY["tool_call_id"], TOOL_NAME, '{"offer_id":"diagnostic-offer:1111111111111111111111111111111111111111111111111111111111111111"}')
        ],
        finish_reason="tool_calls",
        assistant_content="",
        response_observed_at_unix_ms=1_786_000_000_001,
    )
    attestation = consume_provider_tool_invocation(
        **IDENTITY, tool_name=TOOL_NAME
    )
    assert attestation.authorization_id == "permit:slow"


def test_retry_cannot_authorize_an_earlier_tool_call() -> None:
    _authorize()
    retry = {**IDENTITY, "tool_call_id": "call:retry"}
    request_identity = {
        key: retry[key]
        for key in ("task_id", "session_id", "turn_id", "api_request_id")
    }
    begin_provider_request_authorization(
        **request_identity,
        authorization=ProviderRequestAuthorization(
            authorization_id="permit:retry",
            request_sha256="sha256:" + "e" * 64,
            subject_sha256="sha256:" + "f" * 64,
            expires_at_monotonic=time.monotonic() + 10,
        ),
    )
    bind_provider_response_tool_calls(
        **request_identity,
        tool_calls=[
            (retry["tool_call_id"], TOOL_NAME, '{"offer_id":"diagnostic-offer:1111111111111111111111111111111111111111111111111111111111111111"}')
        ],
        finish_reason="tool_calls",
        assistant_content="",
        response_observed_at_unix_ms=1_786_000_000_002,
    )
    with pytest.raises(ProviderRequestBlocked):
        consume_provider_tool_invocation(**IDENTITY, tool_name=TOOL_NAME)
    attestation = consume_provider_tool_invocation(
        **retry, tool_name=TOOL_NAME
    )
    assert attestation.authorization_id == "permit:retry"


def test_duplicate_pre_uniquify_ids_refuse_but_post_uniquify_ids_bind() -> None:
    request_identity = {
        key: IDENTITY[key]
        for key in ("task_id", "session_id", "turn_id", "api_request_id")
    }
    authorization = ProviderRequestAuthorization(
        authorization_id="permit:unique",
        request_sha256="sha256:" + "1" * 64,
        subject_sha256="sha256:" + "2" * 64,
        expires_at_monotonic=time.monotonic() + 10,
    )
    begin_provider_request_authorization(
        **request_identity, authorization=authorization
    )
    with pytest.raises(ProviderRequestBlocked):
        bind_provider_response_tool_calls(
            **request_identity,
            tool_calls=[
                ("call:dup", TOOL_NAME, '{"offer_id":"diagnostic-offer:1111111111111111111111111111111111111111111111111111111111111111"}'),
                ("call:dup", TOOL_NAME, '{"offer_id":"diagnostic-offer:1111111111111111111111111111111111111111111111111111111111111111"}'),
            ],
            finish_reason="tool_calls",
            assistant_content="",
            response_observed_at_unix_ms=1_786_000_000_003,
        )
    discard_provider_request_authorization(**request_identity)
    begin_provider_request_authorization(
        **request_identity, authorization=authorization
    )
    bind_provider_response_tool_calls(
        **request_identity,
        tool_calls=[
            ("call:dup", TOOL_NAME, '{"offer_id":"diagnostic-offer:1111111111111111111111111111111111111111111111111111111111111111"}'),
            ("call:dup:2", TOOL_NAME, '{"offer_id":"diagnostic-offer:1111111111111111111111111111111111111111111111111111111111111111"}'),
        ],
        finish_reason="tool_calls",
        assistant_content="",
        response_observed_at_unix_ms=1_786_000_000_004,
    )
    first = consume_provider_tool_invocation(
        **{**IDENTITY, "tool_call_id": "call:dup"}, tool_name=TOOL_NAME
    )
    second = consume_provider_tool_invocation(
        **{**IDENTITY, "tool_call_id": "call:dup:2"}, tool_name=TOOL_NAME
    )
    assert first.response_binding_sha256 == second.response_binding_sha256


def test_parallel_tool_calls_are_independently_atomic_and_nonreplayable() -> None:
    from concurrent.futures import ThreadPoolExecutor

    request_identity = {
        key: IDENTITY[key]
        for key in ("task_id", "session_id", "turn_id", "api_request_id")
    }
    begin_provider_request_authorization(
        **request_identity,
        authorization=ProviderRequestAuthorization(
            authorization_id="permit:parallel",
            request_sha256="sha256:" + "3" * 64,
            subject_sha256="sha256:" + "4" * 64,
            expires_at_monotonic=time.monotonic() + 10,
        ),
    )
    call_ids = ("call:parallel:1", "call:parallel:2")
    bind_provider_response_tool_calls(
        **request_identity,
        tool_calls=[
            (call_id, TOOL_NAME, '{"offer_id":"diagnostic-offer:1111111111111111111111111111111111111111111111111111111111111111"}')
            for call_id in call_ids
        ],
        finish_reason="tool_calls",
        assistant_content="",
        response_observed_at_unix_ms=1_786_000_000_005,
    )

    def consume(call_id: str):
        return consume_provider_tool_invocation(
            **{**IDENTITY, "tool_call_id": call_id}, tool_name=TOOL_NAME
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(consume, call_ids))
    assert {row.authorization_id for row in results} == {"permit:parallel"}
    for call_id in call_ids:
        with pytest.raises(ProviderRequestBlocked):
            consume(call_id)


def test_wrong_hcp_response_signature_is_never_delivered(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, config, request_key, response_key = _binding(tmp_path)
    monkeypatch.setenv("HCP_DIAGNOSTICS_TOOL_CONFIG", str(config_path))
    _authorize()
    thread, _ = _serve_once(
        Path(config["socket_path"]),
        request_key=request_key,
        response_key=response_key,
        invalid_signature=True,
    )
    result = json.loads(
        hcp_diagnostics_read_tool({"offer_id": "diagnostic-offer:1111111111111111111111111111111111111111111111111111111111111111"}, **IDENTITY)
    )
    thread.join(timeout=3)
    assert result["error_code"] == "BROKER_TRANSPORT_AUTH_INVALID"
