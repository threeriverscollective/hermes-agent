"""Fail-closed authorization boundary for provider requests.

Observer hooks and ordinary middleware intentionally fail open.  A provider
request guard is different: when one is required, exactly one registered guard
must explicitly authorize the immutable request digest before transport runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit


AUTHORIZATION_SCHEMA_VERSION = "hermes.provider-request-authorization.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LOCAL_TRANSPORT_KEYS = frozenset({
    "timeout",
    "http_client",
    "extra_headers",
    "extra_query",
})
_ALLOWED_CLIENT_HEADER_NAMES = frozenset({
    "accept",
    "authorization",
    "content-type",
    "openai-organization",
    "openai-project",
    "user-agent",
    "x-stainless-arch",
    "x-stainless-async",
    "x-stainless-lang",
    "x-stainless-os",
    "x-stainless-package-version",
    "x-stainless-runtime",
    "x-stainless-runtime-version",
})


class ProviderRequestBlocked(BaseException):
    """Fatal authorization refusal that ordinary provider retries cannot catch."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class ProviderRequestGuardRegistrationError(PermissionError):
    """Raised when more than one plugin tries to own provider authorization."""


@dataclass(frozen=True, slots=True)
class ProviderRequestAuthorization:
    """Internal proof that one guard authorized one immutable request digest."""

    authorization_id: str
    request_sha256: str
    subject_sha256: str
    expires_at_monotonic: float
    schema_version: str = AUTHORIZATION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ProviderToolInvocationAttestation:
    """Host-only link from one tool call to its authorized model response."""

    authorization_id: str
    request_sha256: str
    subject_sha256: str
    response_observed_at_unix_ms: int
    response_sha256: str
    response_binding_sha256: str
    binding_receipt_id: str


@dataclass(slots=True)
class _ProviderResponseLedgerEntry:
    authorization: ProviderRequestAuthorization
    task_id: str
    session_id: str
    turn_id: str
    api_request_id: str
    response_observed_at_unix_ms: int | None = None
    response_binding_sha256: str | None = None
    tool_calls: dict[str, str] | None = None


_PROVIDER_RESPONSE_LEDGER: dict[
    tuple[str, str, str, str], _ProviderResponseLedgerEntry
] = {}
_PROVIDER_RESPONSE_LEDGER_LOCK = threading.RLock()
_HCP_DIAGNOSTICS_TOOL = "hcp_diagnostics_read"
_LEDGER_FILENAME = "hcp-provider-response-ledger.sqlite3"


def _ledger_path() -> Path:
    home_text = os.environ.get("HERMES_HOME", "").strip()
    if not home_text:
        raise ProviderRequestBlocked("PROVIDER_RESPONSE_LEDGER_UNAVAILABLE")
    home = Path(home_text)
    if not home.is_absolute() or ".." in home.parts:
        raise ProviderRequestBlocked("PROVIDER_RESPONSE_LEDGER_UNAVAILABLE")
    try:
        info = os.lstat(home)
    except OSError as exc:
        raise ProviderRequestBlocked("PROVIDER_RESPONSE_LEDGER_UNAVAILABLE") from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ProviderRequestBlocked("PROVIDER_RESPONSE_LEDGER_UNAVAILABLE")
    return home / _LEDGER_FILENAME


def _ledger_connect() -> sqlite3.Connection:
    path = _ledger_path()
    try:
        if not path.exists():
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            os.close(descriptor)
        info = os.lstat(path)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise OSError("ledger is not an owner-only regular file")
        connection = sqlite3.connect(str(path), timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS provider_tool_attestations ("
            "task_id TEXT NOT NULL, session_id TEXT NOT NULL, "
            "turn_id TEXT NOT NULL, api_request_id TEXT NOT NULL, "
            "authorization_id TEXT NOT NULL, request_sha256 TEXT NOT NULL, "
            "subject_sha256 TEXT NOT NULL, "
            "response_observed_at_unix_ms INTEGER NOT NULL, "
            "response_sha256 TEXT NOT NULL, "
            "response_binding_sha256 TEXT NOT NULL, "
            "binding_receipt_id TEXT NOT NULL, "
            "tool_calls_json TEXT NOT NULL, "
            "PRIMARY KEY(task_id, session_id, turn_id, api_request_id))"
        )
        return connection
    except (OSError, sqlite3.Error) as exc:
        raise ProviderRequestBlocked("PROVIDER_RESPONSE_LEDGER_UNAVAILABLE") from exc


def _delete_durable_entry(key: tuple[str, str, str, str]) -> None:
    connection = _ledger_connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "DELETE FROM provider_tool_attestations WHERE task_id = ? "
            "AND session_id = ? AND turn_id = ? AND api_request_id = ?",
            key,
        )
        connection.commit()
    except sqlite3.Error as exc:
        connection.rollback()
        raise ProviderRequestBlocked("PROVIDER_RESPONSE_LEDGER_UNAVAILABLE") from exc
    finally:
        connection.close()


def _identity_text(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderRequestBlocked("PROVIDER_REQUEST_CONTEXT_UNAVAILABLE")
    return value


def begin_provider_request_authorization(
    *,
    task_id: object,
    session_id: object,
    turn_id: object,
    api_request_id: object,
    authorization: ProviderRequestAuthorization,
) -> None:
    """Hold one non-secret authorization until its provider response arrives."""

    checked = validate_authorization(
        authorization,
        expected_request_sha256=authorization.request_sha256,
    )
    key = tuple(
        _identity_text(value)
        for value in (task_id, session_id, turn_id, api_request_id)
    )
    assert len(key) == 4
    _delete_durable_entry(key)
    with _PROVIDER_RESPONSE_LEDGER_LOCK:
        _PROVIDER_RESPONSE_LEDGER[key] = _ProviderResponseLedgerEntry(
            authorization=checked,
            task_id=key[0],
            session_id=key[1],
            turn_id=key[2],
            api_request_id=key[3],
        )


def discard_provider_request_authorization(
    *, task_id: object, session_id: object, turn_id: object, api_request_id: object
) -> None:
    """Remove one request that failed before producing a usable response."""

    try:
        key = tuple(
            _identity_text(value)
            for value in (task_id, session_id, turn_id, api_request_id)
        )
    except ProviderRequestBlocked:
        return
    with _PROVIDER_RESPONSE_LEDGER_LOCK:
        _PROVIDER_RESPONSE_LEDGER.pop(key, None)
    try:
        _delete_durable_entry(key)
    except ProviderRequestBlocked:
        return


def bind_provider_response_tool_calls(
    *,
    task_id: object,
    session_id: object,
    turn_id: object,
    api_request_id: object,
    tool_calls: object,
    finish_reason: object,
    assistant_content: object,
    response_observed_at_unix_ms: object,
) -> None:
    """Bind exact normalized HCP tool-call IDs to one authorized response."""

    key = tuple(
        _identity_text(value)
        for value in (task_id, session_id, turn_id, api_request_id)
    )
    if (
        type(response_observed_at_unix_ms) is not int
        or response_observed_at_unix_ms <= 0
        or not isinstance(tool_calls, (tuple, list))
        or type(finish_reason) is not str
        or type(assistant_content) is not str
    ):
        raise ProviderRequestBlocked("PROVIDER_RESPONSE_IDENTITY_INVALID")
    selected: dict[str, str] = {}
    all_calls: list[dict[str, str]] = []
    for item in tool_calls:
        if (
            not isinstance(item, (tuple, list))
            or len(item) != 3
        ):
            raise ProviderRequestBlocked("PROVIDER_RESPONSE_IDENTITY_INVALID")
        tool_call_id = _identity_text(item[0])
        tool_name = _identity_text(item[1])
        if type(item[2]) is not str:
            raise ProviderRequestBlocked("PROVIDER_RESPONSE_IDENTITY_INVALID")
        try:
            arguments = json.loads(item[2])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderRequestBlocked("PROVIDER_RESPONSE_IDENTITY_INVALID") from exc
        try:
            encoded_arguments = json.dumps(
                arguments,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            if json.loads(encoded_arguments) != arguments:
                raise ValueError("arguments do not round-trip")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProviderRequestBlocked(
                "PROVIDER_RESPONSE_IDENTITY_INVALID"
            ) from exc
        arguments_sha256 = (
            "sha256:" + hashlib.sha256(encoded_arguments).hexdigest()
        )
        if any(row["tool_call_id"] == tool_call_id for row in all_calls):
            raise ProviderRequestBlocked("PROVIDER_RESPONSE_IDENTITY_INVALID")
        all_calls.append(
            {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments_sha256": arguments_sha256,
            }
        )
        if tool_name == _HCP_DIAGNOSTICS_TOOL:
            selected[tool_call_id] = tool_name
    with _PROVIDER_RESPONSE_LEDGER_LOCK:
        entry = _PROVIDER_RESPONSE_LEDGER.get(key)
        if entry is None or entry.response_observed_at_unix_ms is not None:
            raise ProviderRequestBlocked("PROVIDER_RESPONSE_UNATTESTED")
        content_sha256 = "sha256:" + hashlib.sha256(
            assistant_content.encode("utf-8")
        ).hexdigest()
        response_sha256 = canonical_request_sha256(
            {
                "finish_reason": finish_reason,
                "content_sha256": content_sha256,
                "tool_calls": all_calls,
            }
        )
        binding = {
            "authorization_id": entry.authorization.authorization_id,
            "request_sha256": entry.authorization.request_sha256,
            "subject_sha256": entry.authorization.subject_sha256,
            "task_id": entry.task_id,
            "session_id": entry.session_id,
            "turn_id": entry.turn_id,
            "api_request_id": entry.api_request_id,
            "response_observed_at_unix_ms": response_observed_at_unix_ms,
            "response_sha256": response_sha256,
            "finish_reason": finish_reason,
            "content_sha256": content_sha256,
            "tool_calls": all_calls,
        }
        entry.response_observed_at_unix_ms = response_observed_at_unix_ms
        from hermes_cli.plugins import bind_provider_response_guard

        receipt = bind_provider_response_guard(**binding)
        response_binding_sha256 = receipt.get("binding_sha256")
        binding_receipt_id = receipt.get("binding_receipt_id")
        if (
            type(response_binding_sha256) is not str
            or _DIGEST.fullmatch(response_binding_sha256) is None
            or type(binding_receipt_id) is not str
            or not binding_receipt_id
            or binding_receipt_id != binding_receipt_id.strip()
        ):
            raise ProviderRequestBlocked("PROVIDER_RESPONSE_BINDING_INVALID")
        entry.response_binding_sha256 = response_binding_sha256
        entry.tool_calls = selected
        if not selected:
            _PROVIDER_RESPONSE_LEDGER.pop(key, None)
            _delete_durable_entry(key)
            return
        durable_calls = json.dumps(
            selected, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        connection = _ledger_connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR REPLACE INTO provider_tool_attestations ("
                "task_id, session_id, turn_id, api_request_id, authorization_id, "
                "request_sha256, subject_sha256, response_observed_at_unix_ms, "
                "response_sha256, response_binding_sha256, binding_receipt_id, "
                "tool_calls_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    *key,
                    entry.authorization.authorization_id,
                    entry.authorization.request_sha256,
                    entry.authorization.subject_sha256,
                    response_observed_at_unix_ms,
                    response_sha256,
                    entry.response_binding_sha256,
                    binding_receipt_id,
                    durable_calls,
                ),
            )
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            raise ProviderRequestBlocked(
                "PROVIDER_RESPONSE_LEDGER_UNAVAILABLE"
            ) from exc
        finally:
            connection.close()
        _PROVIDER_RESPONSE_LEDGER.pop(key, None)


def consume_provider_tool_invocation(
    *,
    task_id: object,
    session_id: object,
    turn_id: object,
    api_request_id: object,
    tool_call_id: object,
    tool_name: object,
) -> ProviderToolInvocationAttestation:
    """Atomically consume one exact tool call from an authorized response."""

    key = tuple(
        _identity_text(value)
        for value in (task_id, session_id, turn_id, api_request_id)
    )
    call_id = _identity_text(tool_call_id)
    name = _identity_text(tool_name)
    if name != _HCP_DIAGNOSTICS_TOOL:
        raise ProviderRequestBlocked("PROVIDER_TOOL_INVOCATION_UNATTESTED")
    connection = _ledger_connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT authorization_id, request_sha256, subject_sha256, "
            "response_observed_at_unix_ms, response_sha256, "
            "response_binding_sha256, binding_receipt_id, tool_calls_json "
            "FROM provider_tool_attestations WHERE task_id = ? "
            "AND session_id = ? AND turn_id = ? AND api_request_id = ?",
            key,
        ).fetchone()
        if row is None:
            raise ProviderRequestBlocked("PROVIDER_TOOL_INVOCATION_UNATTESTED")
        try:
            calls = json.loads(row["tool_calls_json"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderRequestBlocked("PROVIDER_TOOL_INVOCATION_UNATTESTED") from exc
        if type(calls) is not dict or calls.get(call_id) != name:
            raise ProviderRequestBlocked("PROVIDER_TOOL_INVOCATION_UNATTESTED")
        calls.pop(call_id)
        if calls:
            connection.execute(
                "UPDATE provider_tool_attestations SET tool_calls_json = ? "
                "WHERE task_id = ? AND session_id = ? AND turn_id = ? "
                "AND api_request_id = ?",
                (
                    json.dumps(
                        calls,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    *key,
                ),
            )
        else:
            connection.execute(
                "DELETE FROM provider_tool_attestations WHERE task_id = ? "
                "AND session_id = ? AND turn_id = ? AND api_request_id = ?",
                key,
            )
        connection.commit()
        return ProviderToolInvocationAttestation(
            authorization_id=row["authorization_id"],
            request_sha256=row["request_sha256"],
            subject_sha256=row["subject_sha256"],
            response_observed_at_unix_ms=row[
                "response_observed_at_unix_ms"
            ],
            response_sha256=row["response_sha256"],
            response_binding_sha256=row["response_binding_sha256"],
            binding_receipt_id=row["binding_receipt_id"],
        )
    except ProviderRequestBlocked:
        connection.rollback()
        raise
    except sqlite3.Error as exc:
        connection.rollback()
        raise ProviderRequestBlocked("PROVIDER_RESPONSE_LEDGER_UNAVAILABLE") from exc
    finally:
        connection.close()


def canonical_request_sha256(request: Mapping[str, Any]) -> str:
    """Hash one exact JSON provider request without exposing its contents."""

    if not isinstance(request, Mapping) or not request:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_INVALID")
    try:
        encoded = json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if json.loads(encoded) != request:
            raise ValueError("request does not round-trip")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_INVALID") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def canonical_sdk_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return immutable JSON kwargs accepted by the OpenAI SDK call."""

    if not isinstance(request, Mapping):
        raise ProviderRequestBlocked("PROVIDER_REQUEST_INVALID")
    value = {
        key: item for key, item in request.items() if key not in _LOCAL_TRANSPORT_KEYS
    }
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        normalized = json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_INVALID") from exc
    canonical_request_sha256(normalized)
    return normalized


def canonical_model_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact JSON body the OpenAI SDK will put on the wire.

    The SDK merges ``extra_body`` after its ordinary request fields, so those
    values override same-named fields.  Authorization must bind that merged
    body rather than the pre-merge Python kwargs.
    """

    value = canonical_sdk_request(request)
    extra_body = value.pop("extra_body", None)
    if extra_body is not None:
        if not isinstance(extra_body, dict):
            raise ProviderRequestBlocked("PROVIDER_REQUEST_INVALID")
        value.update(extra_body)
    return canonical_sdk_request(value)


def endpoint_origin(base_url: str) -> str:
    """Return a credential-free, path-free origin for permit identity binding."""

    try:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("invalid provider endpoint")
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{parsed.scheme}://{parsed.hostname.lower()}{port}"
    except (TypeError, ValueError) as exc:
        raise ProviderRequestBlocked("PROVIDER_ENDPOINT_INVALID") from exc


def _endpoint_base_url(base_url: object) -> str:
    """Canonicalize one credential-free provider base URL including its path."""

    try:
        parsed = urlsplit(str(base_url))
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("invalid provider endpoint")
        hostname = parsed.hostname.lower()
        if ":" in hostname:
            hostname = f"[{hostname}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        path = parsed.path.rstrip("/") + "/"
        return urlunsplit((parsed.scheme, hostname + port, path, "", ""))
    except (TypeError, ValueError) as exc:
        raise ProviderRequestBlocked("PROVIDER_ENDPOINT_INVALID") from exc


def client_transport_identity(
    client: object,
    *,
    expected_base_url: str,
    expected_api_key: str,
) -> tuple[str, str]:
    """Validate and hash the actual request client's credential-free route.

    Managed HCP traffic supports the SDK's standard headers only and no default
    query parameters.  Header values (including the credential) are hashed
    locally and never disclosed to the permit service.
    """

    expected_endpoint = _endpoint_base_url(expected_base_url)
    actual_endpoint = _endpoint_base_url(getattr(client, "base_url", None))
    if actual_endpoint != expected_endpoint:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_ENDPOINT_MISMATCH")
    default_query = getattr(client, "default_query", None)
    if default_query not in (None, {}) or (
        isinstance(default_query, Mapping) and len(default_query) != 0
    ):
        raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_UNSUPPORTED")
    headers = getattr(client, "default_headers", None)
    if not isinstance(headers, Mapping):
        raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_INVALID")
    normalized_headers: dict[str, str] = {}
    for raw_name, raw_value in headers.items():
        if type(raw_name) is not str:
            raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_INVALID")
        # OpenAI uses a private Omit sentinel for unset organization/project
        # values. It is not sent on the wire and therefore is not part of the
        # transport identity.
        if type(raw_value).__name__ == "Omit":
            continue
        if type(raw_value) is not str:
            raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_INVALID")
        name = raw_name.strip().lower()
        if (
            not name
            or name in normalized_headers
            or name not in _ALLOWED_CLIENT_HEADER_NAMES
        ):
            raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_UNSUPPORTED")
        normalized_headers[name] = raw_value
    if (
        type(expected_api_key) is not str
        or not expected_api_key
        or normalized_headers.get("authorization") != f"Bearer {expected_api_key}"
    ):
        raise ProviderRequestBlocked("PROVIDER_REQUEST_CREDENTIAL_MISMATCH")
    identity = {
        "endpoint_base_url": actual_endpoint,
        "default_headers": normalized_headers,
        "default_query": {},
    }
    return endpoint_origin(actual_endpoint), canonical_request_sha256(identity)


def validate_authorization(
    value: object,
    *,
    expected_request_sha256: str,
) -> ProviderRequestAuthorization:
    """Require the guard's exact typed allow result for the current request."""

    if (
        type(value) is not ProviderRequestAuthorization
        or value.schema_version != AUTHORIZATION_SCHEMA_VERSION
        or not isinstance(value.authorization_id, str)
        or not value.authorization_id
        or value.authorization_id != value.authorization_id.strip()
        or _DIGEST.fullmatch(value.request_sha256) is None
        or value.request_sha256 != expected_request_sha256
        or _DIGEST.fullmatch(value.subject_sha256) is None
        or type(value.expires_at_monotonic) not in {int, float}
        or isinstance(value.expires_at_monotonic, bool)
        or not math.isfinite(float(value.expires_at_monotonic))
        or float(value.expires_at_monotonic) <= time.monotonic()
    ):
        raise ProviderRequestBlocked("PROVIDER_REQUEST_AUTHORIZATION_INVALID")
    return value


def ensure_authorization_current(value: ProviderRequestAuthorization) -> None:
    """Recheck the permit deadline on the final line before provider I/O."""

    if (
        type(value) is not ProviderRequestAuthorization
        or float(value.expires_at_monotonic) <= time.monotonic()
    ):
        raise ProviderRequestBlocked("PROVIDER_REQUEST_AUTHORIZATION_EXPIRED")


__all__ = [
    "AUTHORIZATION_SCHEMA_VERSION",
    "ProviderRequestAuthorization",
    "ProviderRequestBlocked",
    "ProviderRequestGuardRegistrationError",
    "ProviderToolInvocationAttestation",
    "begin_provider_request_authorization",
    "bind_provider_response_tool_calls",
    "client_transport_identity",
    "canonical_request_sha256",
    "canonical_sdk_request",
    "canonical_model_request",
    "endpoint_origin",
    "ensure_authorization_current",
    "consume_provider_tool_invocation",
    "discard_provider_request_authorization",
    "validate_authorization",
]
