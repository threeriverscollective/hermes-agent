from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import threading

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from model_tools import _clear_tool_defs_cache, get_tool_definitions

from tools.hcp_diagnostics_read_tool import (
    CONFIG_SCHEMA_VERSION,
    HCP_DIAGNOSTICS_READ_SCHEMA,
    RESULT_SCHEMA_VERSION,
    WIRE_SCHEMA_VERSION,
    hcp_diagnostics_available,
    hcp_diagnostics_read_tool,
)
from tools.registry import registry
from toolsets import TOOLSETS, _HERMES_CORE_TOOLS


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
    response_sha = _write_private(response_path, response_key.public_key().public_bytes_raw())
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
                unsigned_response = {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "status": "ok",
                    "error_code": None,
                    "repository_id": request["repository_id"],
                    "card_id": request["card_id"],
                    "run_id": request["run_id"],
                    "diagnostics_request_id": request["diagnostics_request_id"],
                    "diagnostics_request_digest": request[
                        "diagnostics_request_digest"
                    ],
                    "nonce": request["nonce"],
                    "records": [{"number": 42, "title": "bounded result"}],
                    "receipt": {"receipt_sha256": "sha256:" + "d" * 64},
                    "replayed": False,
                    "signing_identity": "hcp:broker:repository-a",
                    "key_id": "key:hcp:1",
                }
                signer = Ed25519PrivateKey.generate() if invalid_signature else response_key
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


def test_schema_exposes_only_the_three_opaque_request_fields() -> None:
    parameters = HCP_DIAGNOSTICS_READ_SCHEMA["parameters"]
    assert set(parameters["properties"]) == {
        "diagnostics_request_id",
        "diagnostics_request_digest",
        "nonce",
    }
    assert parameters["additionalProperties"] is False


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


def test_tool_crosses_private_socket_with_host_injected_authority(
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
    thread, observed = _serve_once(
        Path(config["socket_path"]),
        request_key=request_key,
        response_key=response_key,
    )

    result = json.loads(
        hcp_diagnostics_read_tool(
            {
                "diagnostics_request_id": "diag:1",
                "diagnostics_request_digest": "a" * 64,
                "nonce": "nonce:1",
            }
        )
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
    assert request["diagnostics_request_id"] == "diag:1"
    assert request["diagnostics_request_digest"] == "a" * 64
    assert request["nonce"] == "nonce:1"
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before
    } == before


def test_authority_fields_are_rejected_before_socket_access(tmp_path: Path, monkeypatch) -> None:
    config_path, _, _, _ = _binding(tmp_path)
    monkeypatch.setenv("HCP_DIAGNOSTICS_TOOL_CONFIG", str(config_path))
    result = json.loads(
        hcp_diagnostics_read_tool(
            {
                "diagnostics_request_id": "diag:1",
                "diagnostics_request_digest": "a" * 64,
                "nonce": "nonce:1",
                "repository_id": "repository-b",
            }
        )
    )
    assert result["error_code"] == "BROKER_SCOPE_MISMATCH"


def test_wrong_hcp_response_signature_is_never_delivered(tmp_path: Path, monkeypatch) -> None:
    config_path, config, request_key, response_key = _binding(tmp_path)
    monkeypatch.setenv("HCP_DIAGNOSTICS_TOOL_CONFIG", str(config_path))
    thread, _ = _serve_once(
        Path(config["socket_path"]),
        request_key=request_key,
        response_key=response_key,
        invalid_signature=True,
    )
    result = json.loads(
        hcp_diagnostics_read_tool(
            {
                "diagnostics_request_id": "diag:1",
                "diagnostics_request_digest": "a" * 64,
                "nonce": "nonce:1",
            }
        )
    )
    thread.join(timeout=3)
    assert result["error_code"] == "BROKER_TRANSPORT_AUTH_INVALID"
