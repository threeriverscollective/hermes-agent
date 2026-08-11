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
ATTEMPT_RESULT_SCHEMA_VERSION = "hermes.provider-attempt-result.v1"
ATTEMPT_ACK_SCHEMA_VERSION = "hermes.provider-attempt-ack.v1"
MAX_PROVIDER_OUTPUT_BYTES = 100_000_000
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ERROR_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_LOCAL_TRANSPORT_KEYS = frozenset({
    "timeout",
    "http_client",
    "extra_headers",
    "extra_query",
})
_ALLOWED_CLIENT_HEADER_NAMES = frozenset({
    "accept",
    "authorization",
    "chatgpt-account-id",
    "content-type",
    "openai-organization",
    "openai-project",
    "originator",
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
class ProviderAttemptResult:
    """Closed result for the one provider call consumed by an authorization."""

    authorization_id: str
    request_sha256: str
    subject_sha256: str
    outcome: str
    input_tokens: int | None
    output_tokens: int | None
    cache_tokens: int | None
    total_tokens: int | None
    wall_time_ms: int
    stall_time_ms: int
    output_bytes: int
    error_code: str | None
    schema_version: str = ATTEMPT_RESULT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ProviderAttemptAcknowledgment:
    """Typed proof that HCP durably recorded one exact provider attempt."""

    authorization_id: str
    request_sha256: str
    subject_sha256: str
    usage_receipt_sha256: str
    schema_version: str = ATTEMPT_ACK_SCHEMA_VERSION


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
    expected_client_headers: Mapping[str, str] | None = None,
    request_headers: object = None,
    expected_request_headers: Mapping[str, str] | None = None,
    request_query: object = None,
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
    http_client = getattr(client, "_client", None)
    if getattr(http_client, "follow_redirects", None) is not False:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_UNSUPPORTED")
    if getattr(client, "max_retries", None) != 0:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_UNSUPPORTED")
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
    normalized_expected_client_headers: dict[str, str] = {}
    for raw_name, raw_value in (expected_client_headers or {}).items():
        if type(raw_name) is not str or type(raw_value) is not str:
            raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_INVALID")
        name = raw_name.strip().lower()
        if (
            not name
            or name in normalized_expected_client_headers
            or name not in _ALLOWED_CLIENT_HEADER_NAMES
        ):
            raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_INVALID")
        normalized_expected_client_headers[name] = raw_value
    managed_client_headers = frozenset({"originator", "chatgpt-account-id"})
    if any(
        normalized_headers.get(name) != value
        for name, value in normalized_expected_client_headers.items()
    ) or any(
        name in normalized_headers and name not in normalized_expected_client_headers
        for name in managed_client_headers
    ):
        raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_UNSUPPORTED")
    if request_query not in (None, {}) or (
        isinstance(request_query, Mapping) and len(request_query) != 0
    ):
        raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_UNSUPPORTED")
    expected_headers = dict(expected_request_headers or {})
    if request_headers is None:
        actual_request_headers: dict[str, str] = {}
    elif isinstance(request_headers, Mapping):
        actual_request_headers = {}
        for raw_name, raw_value in request_headers.items():
            if type(raw_name) is not str or type(raw_value) is not str:
                raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_INVALID")
            name = raw_name.strip().lower()
            if not name or name in actual_request_headers:
                raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_INVALID")
            actual_request_headers[name] = raw_value
    else:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_INVALID")
    if actual_request_headers != expected_headers:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_TRANSPORT_UNSUPPORTED")
    identity = {
        "endpoint_base_url": actual_endpoint,
        "follow_redirects": False,
        "sdk_max_retries": 0,
        "default_headers": normalized_headers,
        "default_query": {},
        "request_headers": actual_request_headers,
        "request_query": {},
    }
    return endpoint_origin(actual_endpoint), canonical_request_sha256(identity)


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _exact_nonnegative_int(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_USAGE_INVALID")
    return value


def exact_provider_usage(
    response: object, *, api_mode: str
) -> tuple[int, int, int, int]:
    """Extract exact provider-reported input/output/cache/total token counts."""

    usage = _field(response, "usage")
    if usage is None:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_USAGE_INVALID")
    if api_mode == "codex_responses":
        input_tokens = _exact_nonnegative_int(_field(usage, "input_tokens"))
        output_tokens = _exact_nonnegative_int(_field(usage, "output_tokens"))
        details = _field(usage, "input_tokens_details")
    elif api_mode == "chat_completions":
        input_tokens = _exact_nonnegative_int(_field(usage, "prompt_tokens"))
        output_tokens = _exact_nonnegative_int(_field(usage, "completion_tokens"))
        details = _field(usage, "prompt_tokens_details")
    else:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_ROUTE_UNSUPPORTED")
    total_tokens = _exact_nonnegative_int(_field(usage, "total_tokens"))
    cache_tokens = _exact_nonnegative_int(_field(details, "cached_tokens"))
    if cache_tokens > input_tokens or total_tokens != input_tokens + output_tokens:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_USAGE_INVALID")
    return input_tokens, output_tokens, cache_tokens, total_tokens


def exact_serialized_provider_output(response: object) -> bytes:
    """Return the exact bounded canonical JSON representation of one response."""

    if isinstance(response, Mapping):
        value = dict(response)
    else:
        dump = getattr(response, "model_dump", None)
        if not callable(dump):
            raise ProviderRequestBlocked("PROVIDER_REQUEST_OUTPUT_INVALID")
        try:
            value = dump(mode="json")
        except Exception as exc:
            raise ProviderRequestBlocked("PROVIDER_REQUEST_OUTPUT_INVALID") from exc
    if not isinstance(value, Mapping):
        raise ProviderRequestBlocked("PROVIDER_REQUEST_OUTPUT_INVALID")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if json.loads(encoded) != value:
            raise ValueError("response does not round-trip")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_OUTPUT_INVALID") from exc
    if len(encoded) > MAX_PROVIDER_OUTPUT_BYTES:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_OUTPUT_TOO_LARGE")
    return encoded


def validate_attempt_result(value: object) -> ProviderAttemptResult:
    if type(value) is not ProviderAttemptResult:
        raise ProviderRequestBlocked("PROVIDER_REQUEST_COMPLETION_INVALID")
    token_values = (
        value.input_tokens,
        value.output_tokens,
        value.cache_tokens,
        value.total_tokens,
    )
    valid_success = (
        value.outcome == "SUCCESS"
        and value.error_code is None
        and all(type(item) is int and item >= 0 for item in token_values)
        and value.cache_tokens <= value.input_tokens
        and value.total_tokens == value.input_tokens + value.output_tokens
    )
    valid_error = (
        value.outcome == "PROVIDER_ERROR"
        and all(item is None for item in token_values)
        and type(value.error_code) is str
        and _ERROR_CODE.fullmatch(value.error_code) is not None
    )
    if (
        value.schema_version != ATTEMPT_RESULT_SCHEMA_VERSION
        or not value.authorization_id
        or _DIGEST.fullmatch(value.request_sha256) is None
        or _DIGEST.fullmatch(value.subject_sha256) is None
        or not (valid_success or valid_error)
        or type(value.wall_time_ms) is not int
        or value.wall_time_ms < 0
        or type(value.stall_time_ms) is not int
        or not 0 <= value.stall_time_ms <= value.wall_time_ms
        or type(value.output_bytes) is not int
        or not 0 <= value.output_bytes <= MAX_PROVIDER_OUTPUT_BYTES
    ):
        raise ProviderRequestBlocked("PROVIDER_REQUEST_COMPLETION_INVALID")
    return value


def validate_attempt_acknowledgment(
    value: object, *, expected: ProviderAttemptResult
) -> ProviderAttemptAcknowledgment:
    if (
        type(value) is not ProviderAttemptAcknowledgment
        or value.schema_version != ATTEMPT_ACK_SCHEMA_VERSION
        or value.authorization_id != expected.authorization_id
        or value.request_sha256 != expected.request_sha256
        or value.subject_sha256 != expected.subject_sha256
        or _DIGEST.fullmatch(value.usage_receipt_sha256) is None
    ):
        raise ProviderRequestBlocked("PROVIDER_REQUEST_COMPLETION_INVALID")
    return value


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
    "ATTEMPT_ACK_SCHEMA_VERSION",
    "ATTEMPT_RESULT_SCHEMA_VERSION",
    "AUTHORIZATION_SCHEMA_VERSION",
    "MAX_PROVIDER_OUTPUT_BYTES",
    "ProviderAttemptAcknowledgment",
    "ProviderAttemptResult",
    "ProviderRequestAuthorization",
    "ProviderRequestBlocked",
    "ProviderRequestGuardRegistrationError",
    "client_transport_identity",
    "canonical_request_sha256",
    "canonical_sdk_request",
    "canonical_model_request",
    "endpoint_origin",
    "exact_provider_usage",
    "exact_serialized_provider_output",
    "ensure_authorization_current",
    "validate_authorization",
    "validate_attempt_acknowledgment",
    "validate_attempt_result",
]
