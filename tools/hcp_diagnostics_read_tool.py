"""Pinned host-side bridge for HCP repository-scoped diagnostic reads.

This tool is intentionally unavailable unless an HCP-launched Hermes process
receives one owner-only, exact runtime binding.  The model supplies only one
stable opaque offer id. HCP materializes the fresh current-permit-bound request
and nonce at host-side invocation. Repository/card/run identities and both
transport keys come from that binding, and the Docker terminal never receives
the socket or key paths.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import struct
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from tools.registry import registry, tool_error


TOOL_NAME = "hcp_diagnostics_read"
WIRE_SCHEMA_VERSION = "hermes.hcp.diagnostics-read-wire.v2"
RESULT_SCHEMA_VERSION = "hermes.hcp.diagnostics-read-result.v2"
CONFIG_SCHEMA_VERSION = "hermes.hcp.diagnostics-tool-config.v1"
_CONFIG_ENV = "HCP_DIAGNOSTICS_TOOL_CONFIG"
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_INPUT_KEYS = frozenset({"offer_id"})
_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "repository_id",
        "card_id",
        "run_id",
        "socket_path",
        "request_signing",
        "response_verification",
        "timeout_ms",
    }
)
_KEY_KEYS = frozenset(
    {"signing_identity", "key_id", "key_path", "key_sha256"}
)
_RESPONSE_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "error_code",
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
        "records",
        "receipt",
        "replayed",
        "signing_identity",
        "key_id",
        "signature",
    }
)


class HcpDiagnosticsToolRefusal(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _json_value(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        if isinstance(value, str) and any(ord(character) < 0x20 for character in value):
            raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
        return {key: _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            _json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID") from error


def _strict_json(raw: bytes, *, limit: int) -> dict[str, object]:
    if not 1 <= len(raw) <= limit:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except HcpDiagnosticsToolRefusal:
        raise
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID") from error
    if type(value) is not dict or _canonical(value) != raw:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    return value


def _token(value: object) -> str:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    return value


def _read_owner_file(path_value: object) -> tuple[Path, bytes]:
    if type(path_value) is not str:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    path = Path(path_value)
    if not path.is_absolute() or ".." in path.parts:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        payload = b""
        while len(payload) <= _MAX_REQUEST_BYTES:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            payload += chunk
    except OSError as error:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID") from error
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or len(payload) > _MAX_REQUEST_BYTES
    ):
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    return path, payload


def _read_private_file(path_value: object, expected_digest: object) -> bytes:
    if type(expected_digest) is not str or _SHA256.fullmatch(expected_digest) is None:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    _, payload = _read_owner_file(path_value)
    if "sha256:" + hashlib.sha256(payload).hexdigest() != expected_digest:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    return payload


def _load_binding() -> dict[str, object]:
    config_path = os.environ.get(_CONFIG_ENV, "")
    if not config_path:
        raise HcpDiagnosticsToolRefusal("DIAGNOSTICS_NOT_ADMITTED")
    config, raw = _read_owner_file(config_path)
    parent = config.parent
    parent_metadata = os.lstat(parent)
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or stat.S_ISLNK(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
    ):
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    value = _strict_json(raw, limit=_MAX_REQUEST_BYTES)
    if set(value) != _CONFIG_KEYS or value["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    for field in ("repository_id", "card_id", "run_id"):
        _token(value[field])
    socket_path = value["socket_path"]
    if (
        type(socket_path) is not str
        or not Path(socket_path).is_absolute()
        or Path(socket_path).parent != parent
    ):
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    timeout_ms = value["timeout_ms"]
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 30_000:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    for field in ("request_signing", "response_verification"):
        row = value[field]
        if type(row) is not dict or set(row) != _KEY_KEYS:
            raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
        _token(row["signing_identity"])
        _token(row["key_id"])
        if Path(str(row["key_path"])).parent != parent:
            raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
        _read_private_file(row["key_path"], row["key_sha256"])
    return value


def hcp_diagnostics_available() -> bool:
    try:
        _load_binding()
    except HcpDiagnosticsToolRefusal:
        return False
    return True


def _read_exact(channel: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = channel.recv(remaining)
        if not chunk:
            raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_UNAVAILABLE")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(channel: socket.socket) -> dict[str, object]:
    size = struct.unpack(">I", _read_exact(channel, 4))[0]
    if not 1 <= size <= _MAX_RESPONSE_BYTES:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    return _strict_json(_read_exact(channel, size), limit=_MAX_RESPONSE_BYTES)


def _write_frame(channel: socket.socket, value: Mapping[str, object]) -> None:
    payload = _canonical(value)
    if len(payload) > _MAX_REQUEST_BYTES:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    channel.sendall(struct.pack(">I", len(payload)) + payload)


def _execute(
    value: Mapping[str, object],
    *,
    task_id: object,
    session_id: object,
    turn_id: object,
    api_request_id: object,
    tool_call_id: object,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _INPUT_KEYS:
        raise HcpDiagnosticsToolRefusal("BROKER_SCOPE_MISMATCH")
    binding = _load_binding()
    signing = binding["request_signing"]
    verification = binding["response_verification"]
    assert isinstance(signing, dict) and isinstance(verification, dict)
    private_key = Ed25519PrivateKey.from_private_bytes(
        _read_private_file(signing["key_path"], signing["key_sha256"])
    )
    public_key = Ed25519PublicKey.from_public_bytes(
        _read_private_file(verification["key_path"], verification["key_sha256"])
    )
    invocation = {
        "task_id": _token(task_id),
        "session_id": _token(session_id),
        "turn_id": _token(turn_id),
        "api_request_id": _token(api_request_id),
        "tool_call_id": _token(tool_call_id),
    }
    if invocation["task_id"] != binding["card_id"]:
        raise HcpDiagnosticsToolRefusal("BROKER_SCOPE_MISMATCH")
    from hermes_cli.provider_request_guard import (
        ProviderRequestBlocked,
        consume_provider_tool_invocation,
    )

    try:
        attestation = consume_provider_tool_invocation(
            **invocation,
            tool_name=TOOL_NAME,
        )
    except ProviderRequestBlocked as error:
        raise HcpDiagnosticsToolRefusal(
            "BROKER_TRANSPORT_AUTH_INVALID"
        ) from error
    provider_attestation = {
        "provider_authorization_id": _token(attestation.authorization_id),
        "provider_request_sha256": attestation.request_sha256,
        "provider_subject_sha256": attestation.subject_sha256,
        "provider_response_observed_at_unix_ms": (
            attestation.response_observed_at_unix_ms
        ),
        "provider_response_sha256": attestation.response_sha256,
        "provider_response_binding_sha256": (
            attestation.response_binding_sha256
        ),
        "provider_response_binding_receipt_id": attestation.binding_receipt_id,
    }
    if (
        _SHA256.fullmatch(provider_attestation["provider_request_sha256"])
        is None
        or _SHA256.fullmatch(provider_attestation["provider_subject_sha256"])
        is None
        or _SHA256.fullmatch(
            provider_attestation["provider_response_sha256"]
        )
        is None
        or _SHA256.fullmatch(
            provider_attestation["provider_response_binding_sha256"]
        )
        is None
        or type(
            provider_attestation["provider_response_binding_receipt_id"]
        )
        is not str
        or not provider_attestation["provider_response_binding_receipt_id"]
        or type(
            provider_attestation["provider_response_observed_at_unix_ms"]
        )
        is not int
        or provider_attestation["provider_response_observed_at_unix_ms"] <= 0
    ):
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    unsigned = {
        "schema_version": WIRE_SCHEMA_VERSION,
        "operation": TOOL_NAME,
        "repository_id": binding["repository_id"],
        "card_id": binding["card_id"],
        "run_id": binding["run_id"],
        "offer_id": _token(value["offer_id"]),
        **invocation,
        **provider_attestation,
        "signing_identity": signing["signing_identity"],
        "key_id": signing["key_id"],
    }
    request = {**unsigned, "signature": private_key.sign(_canonical(unsigned)).hex()}
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    channel.settimeout(int(binding["timeout_ms"]) / 1000)
    try:
        channel.connect(str(binding["socket_path"]))
        _write_frame(channel, request)
        response = _read_frame(channel)
    except (OSError, socket.timeout) as error:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_UNAVAILABLE") from error
    finally:
        channel.close()
    if set(response) != _RESPONSE_KEYS:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    signature = response["signature"]
    if type(signature) is not str or len(signature) != 128:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    response_unsigned = dict(response)
    response_unsigned.pop("signature")
    try:
        public_key.verify(bytes.fromhex(signature), _canonical(response_unsigned))
    except (InvalidSignature, ValueError) as error:
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID") from error
    expected = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "repository_id": binding["repository_id"],
        "card_id": binding["card_id"],
        "run_id": binding["run_id"],
        "offer_id": value["offer_id"],
        **invocation,
        **provider_attestation,
        "signing_identity": verification["signing_identity"],
        "key_id": verification["key_id"],
    }
    if any(response.get(key) != expected_value for key, expected_value in expected.items()):
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    status = response["status"]
    error_code = response["error_code"]
    if status == "refused" and type(error_code) is str:
        raise HcpDiagnosticsToolRefusal(error_code)
    if (
        status != "ok"
        or error_code is not None
        or type(response["records"]) is not list
        or type(response["receipt"]) is not dict
        or type(response["replayed"]) is not bool
    ):
        raise HcpDiagnosticsToolRefusal("BROKER_TRANSPORT_AUTH_INVALID")
    return {
        "status": "ok",
        "records": response["records"],
        "receipt": response["receipt"],
        "replayed": response["replayed"],
    }


def hcp_diagnostics_read_tool(
    args: object,
    *,
    task_id: object = "",
    session_id: object = "",
    turn_id: object = "",
    api_request_id: object = "",
    tool_call_id: object = "",
) -> str:
    try:
        result = _execute(
            args,  # type: ignore[arg-type]
            task_id=task_id,
            session_id=session_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            tool_call_id=tool_call_id,
        )
    except HcpDiagnosticsToolRefusal as error:
        return tool_error(error.code, error_code=error.code)
    return json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


HCP_DIAGNOSTICS_READ_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Read one HCP-admitted, repository-scoped diagnostic offer. The stable "
        "opaque offer id must exactly match the current task context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "offer_id": {"type": "string"},
        },
        "required": ["offer_id"],
        "additionalProperties": False,
    },
}


registry.register(
    name=TOOL_NAME,
    toolset="hcp_diagnostics",
    schema=HCP_DIAGNOSTICS_READ_SCHEMA,
    handler=lambda args, **identity: hcp_diagnostics_read_tool(
        args,
        task_id=identity.get("task_id", ""),
        session_id=identity.get("session_id", ""),
        turn_id=identity.get("turn_id", ""),
        api_request_id=identity.get("api_request_id", ""),
        tool_call_id=identity.get("tool_call_id", ""),
    ),
    check_fn=hcp_diagnostics_available,
    emoji="🔎",
    max_result_size_chars=2 * 1024 * 1024,
)
