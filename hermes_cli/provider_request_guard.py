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
import re
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
    "client_transport_identity",
    "canonical_request_sha256",
    "canonical_sdk_request",
    "canonical_model_request",
    "endpoint_origin",
    "ensure_authorization_current",
    "validate_authorization",
]
