"""HCP post-claim, pre-model permit plugin.

Hermes still owns claim, profile selection, scheduling, and worker launch. This
plugin only authenticates one already-claimed worker to the HCP permit server,
then requires HCP to issue and consume one request-bound permit immediately
before Hermes performs the provider call.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import re
import secrets
import stat
import threading
import time
from typing import Callable, Mapping

try:  # POSIX-only capability descriptor checks; ordinary Windows stays inert.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows CI
    fcntl = None

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from hermes_cli.provider_request_guard import (
    ProviderRequestAuthorization,
    ProviderRequestBlocked,
)

from .channel import UnixSocketPermitTransport, canonical_json


MANIFEST_SCHEMA_VERSION = "hermes.hcp.pre-model-permit-client.v2"
WIRE_SCHEMA_VERSION = "hermes.hcp.pre-model-permit-wire.v1"
RESULT_SCHEMA_VERSION = "hermes.hcp.pre-model-permit-result.v1"
SUBJECT_SCHEMA_VERSION = "hermes.hcp.worker-subject.v1"
PERMIT_SCHEMA_VERSION = "hcp.pre-model-permit.v2"
PERMIT_EXCHANGE_SCHEMA_VERSION = "hcp.hermes.permit-exchange.v1"

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SOURCE_OID = re.compile(r"[0-9a-f]{40,64}\Z")
_SIGNATURE = re.compile(r"[0-9a-f]{128}\Z")
_ERROR_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_NONCE = re.compile(r"[0-9a-f]{64}\Z")
_MAX_MANIFEST_BYTES = 256 * 1024

_SNAPSHOT_KEYS = {
    "schema_version",
    "card_id",
    "bead_id",
    "item_revision",
    "repository_id",
    "protected_source_revision",
    "authority_digest",
    "binding_digest",
    "scope_digest",
    "dependency_digest",
    "collision_digest",
    "revocation_digest",
    "run_id",
    "generation_id",
    "selected_profile_id",
    "allowed_profile_ids",
    "required_capability_classes",
    "profile_capability_classes",
    "auth_mode",
    "budget_enforcement_mode",
    "model_budget_remaining",
    "retry_budget_remaining",
    "budget_policy_digest",
}
_MANIFEST_KEYS = {
    "schema_version",
    "task_id",
    "generation_id",
    "board_id",
    "profile_id",
    "binding_digest",
    "claim_lock_sha256",
    "provider",
    "model",
    "api_mode",
    "endpoint_origin",
    "reasoning_effort",
    "transport_mode",
    "peer_identity",
    "peer_key_id",
    "server_identity",
    "server_key_id",
    "permit_ttl_seconds",
    "model_tokens_per_request",
    "snapshot",
}
_PERMIT_KEYS = {
    "schema_version",
    "permit_id",
    "request_digest",
    "snapshot",
    "auth_mode",
    "budget_enforcement_mode",
    "nonce",
    "model_token_limit",
    "retry_budget_charge",
    "issued_at",
    "expires_at",
    "canonical_model_request_digest",
    "signing_identity",
    "key_id",
    "signature",
}
_RESULT_KEYS = {
    "schema_version",
    "operation",
    "subject_sha256",
    "permit_exchange_nonce",
    "verify_exchange_nonce",
    "outcome",
    "error_code",
    "permit",
    "verification",
    "server_identity",
    "server_key_id",
    "response_nonce",
    "issued_at",
    "signature",
}


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def _text(value: object, error_code: str = "HCP_PERMIT_RESPONSE_INVALID") -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 4096
        or any(ord(char) < 0x20 for char in value)
    ):
        raise ProviderRequestBlocked(error_code)
    return value


def _timestamp(
    value: object, error_code: str = "HCP_PERMIT_RESPONSE_INVALID"
) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ProviderRequestBlocked(error_code)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ProviderRequestBlocked(error_code) from exc
    if parsed.tzinfo != timezone.utc:
        raise ProviderRequestBlocked(error_code)
    return parsed


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo != timezone.utc:
        raise ProviderRequestBlocked("HCP_PERMIT_CLOCK_INVALID")
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _strict_json(encoded: bytes, *, error_code: str) -> dict[str, object]:
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProviderRequestBlocked(error_code) from exc
    if not isinstance(value, dict) or canonical_json(value) != encoded:
        raise ProviderRequestBlocked(error_code)
    return value


def _validate_snapshot(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _SNAPSHOT_KEYS:
        raise ProviderRequestBlocked("HCP_PERMIT_MANIFEST_INVALID")
    snapshot = dict(value)
    if snapshot.get("schema_version") != PERMIT_SCHEMA_VERSION:
        raise ProviderRequestBlocked("HCP_PERMIT_MANIFEST_INVALID")
    for key in (
        "card_id",
        "bead_id",
        "item_revision",
        "repository_id",
        "run_id",
        "generation_id",
        "selected_profile_id",
    ):
        _text(snapshot.get(key), "HCP_PERMIT_MANIFEST_INVALID")
    if _SOURCE_OID.fullmatch(str(snapshot.get("protected_source_revision"))) is None:
        raise ProviderRequestBlocked("HCP_PERMIT_MANIFEST_INVALID")
    for key in (
        "authority_digest",
        "binding_digest",
        "scope_digest",
        "dependency_digest",
        "collision_digest",
        "revocation_digest",
        "budget_policy_digest",
    ):
        if _DIGEST.fullmatch(str(snapshot.get(key))) is None:
            raise ProviderRequestBlocked("HCP_PERMIT_MANIFEST_INVALID")
    for key in (
        "allowed_profile_ids",
        "required_capability_classes",
        "profile_capability_classes",
    ):
        items = snapshot.get(key)
        if not isinstance(items, list) or not items or len(items) != len(set(items)):
            raise ProviderRequestBlocked("HCP_PERMIT_MANIFEST_INVALID")
        for item in items:
            _text(item, "HCP_PERMIT_MANIFEST_INVALID")
    auth_mode = snapshot.get("auth_mode")
    budget_mode = snapshot.get("budget_enforcement_mode")
    if (
        auth_mode not in {"API", "OAUTH_OBSERVATIONAL"}
        or budget_mode != ("ENFORCED" if auth_mode == "API" else "OBSERVATIONAL")
        or type(snapshot.get("model_budget_remaining")) is not int
        or type(snapshot.get("retry_budget_remaining")) is not int
    ):
        raise ProviderRequestBlocked("HCP_PERMIT_MANIFEST_INVALID")
    return snapshot


def _validate_manifest(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _MANIFEST_KEYS:
        raise ProviderRequestBlocked("HCP_PERMIT_MANIFEST_INVALID")
    manifest = dict(value)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ProviderRequestBlocked("HCP_PERMIT_MANIFEST_INVALID")
    for key in (
        "task_id",
        "generation_id",
        "board_id",
        "profile_id",
        "provider",
        "model",
        "api_mode",
        "endpoint_origin",
        "peer_identity",
        "peer_key_id",
        "server_identity",
        "server_key_id",
    ):
        _text(manifest.get(key), "HCP_PERMIT_MANIFEST_INVALID")
    for key in ("binding_digest", "claim_lock_sha256"):
        if _DIGEST.fullmatch(str(manifest.get(key))) is None:
            raise ProviderRequestBlocked("HCP_PERMIT_MANIFEST_INVALID")
    if (
        manifest["api_mode"] not in {"chat_completions", "codex_responses"}
        or manifest.get("transport_mode") != "non_streaming"
        or (
            manifest["api_mode"] == "chat_completions"
            and manifest.get("reasoning_effort") is not None
        )
        or (
            manifest["api_mode"] == "codex_responses"
            and (
                manifest.get("provider") != "openai-codex"
                or type(manifest.get("reasoning_effort")) is not str
                or not str(manifest["reasoning_effort"])
                or manifest["reasoning_effort"]
                != str(manifest["reasoning_effort"]).strip()
            )
        )
        or type(manifest.get("permit_ttl_seconds")) is not int
        or not 1 <= manifest["permit_ttl_seconds"] <= 30
        or type(manifest.get("model_tokens_per_request")) is not int
        or manifest["model_tokens_per_request"] <= 0
    ):
        raise ProviderRequestBlocked("HCP_PERMIT_MANIFEST_INVALID")
    snapshot = _validate_snapshot(manifest.get("snapshot"))
    if (
        manifest["task_id"] != snapshot["card_id"]
        or manifest["generation_id"] != snapshot["generation_id"]
        or manifest["profile_id"] != snapshot["selected_profile_id"]
        or manifest["profile_id"] not in snapshot["allowed_profile_ids"]
        or manifest["binding_digest"] != snapshot["binding_digest"]
        or manifest["model_tokens_per_request"] > snapshot["model_budget_remaining"]
    ):
        raise ProviderRequestBlocked("HCP_PERMIT_MANIFEST_MISMATCH")
    manifest["snapshot"] = snapshot
    return manifest


def _read_protected_fd(env_name: str, *, limit: int) -> bytes:
    fd_text = os.environ.pop(env_name, "").strip()
    fd: int | None = None
    if fcntl is None or not fd_text.isdecimal() or int(fd_text) < 3:
        raise ProviderRequestBlocked("HCP_PERMIT_WORKER_INPUT_INVALID")
    fd = int(fd_text)
    try:
        before = os.fstat(fd)
        access = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o400
            or access != os.O_RDONLY
            or not 1 <= before.st_size <= limit
        ):
            raise OSError("invalid protected input")
        chunks = bytearray()
        offset = 0
        while len(chunks) < before.st_size:
            chunk = os.pread(fd, before.st_size - len(chunks), offset)
            if not chunk:
                break
            chunks.extend(chunk)
            offset += len(chunk)
        after = os.fstat(fd)
        if len(chunks) != before.st_size or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise OSError("protected input changed")
        return bytes(chunks)
    except OSError as exc:
        raise ProviderRequestBlocked("HCP_PERMIT_WORKER_INPUT_INVALID") from exc
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _close_remaining_worker_inputs() -> None:
    """Close every capability FD even when an earlier input was malformed."""
    seen: set[int] = set()
    for name in (
        "HCP_PRE_MODEL_PERMIT_MANIFEST_FD",
        "HCP_PRE_MODEL_PERMIT_PEER_PRIVATE_KEY_FD",
        "HCP_PRE_MODEL_PERMIT_SERVER_PUBLIC_KEY_FD",
    ):
        text = os.environ.pop(name, "").strip()
        if not text.isdecimal():
            continue
        fd = int(text)
        if fd < 3 or fd in seen:
            continue
        seen.add(fd)
        try:
            os.close(fd)
        except OSError:
            pass


def _sign(private_key: Ed25519PrivateKey, value: object) -> str:
    return private_key.sign(canonical_json(value)).hex()


def _verify(public_key: Ed25519PublicKey, value: object, signature: object) -> None:
    if type(signature) is not str or _SIGNATURE.fullmatch(signature) is None:
        raise ProviderRequestBlocked("HCP_PERMIT_SIGNATURE_INVALID")
    try:
        public_key.verify(bytes.fromhex(signature), canonical_json(value))
    except (InvalidSignature, ValueError) as exc:
        raise ProviderRequestBlocked("HCP_PERMIT_SIGNATURE_INVALID") from exc


def _exchange(
    *,
    request: Mapping[str, object],
    operation: str,
    sender_identity: str,
    receiver_identity: str,
    key_id: str,
    nonce: str,
    sequence: int,
    private_key: Ed25519PrivateKey,
) -> dict[str, object]:
    auth_payload = {
        "schema_version": PERMIT_EXCHANGE_SCHEMA_VERSION,
        "operation": operation,
        "sender_identity": sender_identity,
        "receiver_identity": receiver_identity,
        "key_id": key_id,
        "nonce": nonce,
        "sequence": sequence,
        "body": dict(request),
    }
    return {
        "schema_version": PERMIT_EXCHANGE_SCHEMA_VERSION,
        "operation": operation,
        "request": dict(request),
        "sender_identity": sender_identity,
        "receiver_identity": receiver_identity,
        "key_id": key_id,
        "nonce": nonce,
        "sequence": sequence,
        "signature": _sign(private_key, auth_payload),
    }


class HCPPermitGuard:
    """Obtain and consume one HCP permit for each imminent provider call."""

    def __init__(
        self,
        *,
        transport: UnixSocketPermitTransport,
        manifest: Mapping[str, object],
        peer_private_key: Ed25519PrivateKey,
        server_public_key: Ed25519PublicKey,
        hermes_run_id: str,
        nonce_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._transport = transport
        self._manifest = _validate_manifest(manifest)
        self._peer_private_key = peer_private_key
        self._server_public_key = server_public_key
        if not hermes_run_id.isdecimal() or int(hermes_run_id) <= 0:
            raise ProviderRequestBlocked("HCP_PERMIT_IDENTITY_INVALID")
        self._hermes_run_id = hermes_run_id
        self._nonce_factory = nonce_factory or (lambda: secrets.token_hex(32))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sequence = 0
        self._session_id: str | None = None
        self._used_nonces: set[str] = set()
        self._used_permit_ids: set[str] = set()
        self._lock = threading.Lock()

    def _nonce(self) -> str:
        value = self._nonce_factory()
        if type(value) is not str or _NONCE.fullmatch(value) is None:
            raise ProviderRequestBlocked("HCP_PERMIT_NONCE_INVALID")
        if value in self._used_nonces:
            raise ProviderRequestBlocked("HCP_PERMIT_NONCE_REPLAY")
        self._used_nonces.add(value)
        return value

    def __call__(
        self,
        *,
        request: Mapping[str, object],
        request_sha256: str,
        task_id: str,
        turn_id: str,
        api_request_id: str,
        session_id: str,
        profile_id: str,
        provider: str,
        model: str,
        api_mode: str,
        endpoint_origin: str,
        transport_identity_sha256: str,
        transport_mode: str,
        api_call_count: int,
        model_tokens_requested: int | None,
        **_: object,
    ) -> ProviderRequestAuthorization:
        manifest = self._manifest
        if (
            _DIGEST.fullmatch(request_sha256) is None
            or request_sha256 != _digest(request)
            or task_id != manifest["task_id"]
            or profile_id != manifest["profile_id"]
            or provider != manifest["provider"]
            or model != manifest["model"]
            or api_mode != manifest["api_mode"]
            or endpoint_origin != manifest["endpoint_origin"]
            or _DIGEST.fullmatch(transport_identity_sha256) is None
            or transport_mode != manifest["transport_mode"]
            or request.get("model") != model
            or type(api_call_count) is not int
            or api_call_count < 0
        ):
            raise ProviderRequestBlocked("HCP_PERMIT_IDENTITY_MISMATCH")
        if api_mode == "codex_responses":
            reasoning = request.get("reasoning")
            if (
                not isinstance(reasoning, Mapping)
                or reasoning.get("effort") != manifest["reasoning_effort"]
            ):
                raise ProviderRequestBlocked("HCP_PERMIT_IDENTITY_MISMATCH")
        for value in (turn_id, api_request_id, session_id):
            _text(value, "HCP_PERMIT_IDENTITY_MISMATCH")
        requested_tokens = (
            manifest["model_tokens_per_request"]
            if model_tokens_requested is None
            else model_tokens_requested
        )
        if (
            type(requested_tokens) is not int
            or requested_tokens <= 0
            or requested_tokens > manifest["model_tokens_per_request"]
        ):
            raise ProviderRequestBlocked("HCP_PERMIT_MODEL_BUDGET_INVALID")

        with self._lock:
            if self._session_id is None:
                self._session_id = session_id
            elif self._session_id != session_id:
                raise ProviderRequestBlocked("HCP_PERMIT_IDENTITY_MISMATCH")
            now = self._clock()
            if now.tzinfo != timezone.utc:
                raise ProviderRequestBlocked("HCP_PERMIT_CLOCK_INVALID")
            issued_at = _format_timestamp(now)
            expires_at = _format_timestamp(
                now + timedelta(seconds=manifest["permit_ttl_seconds"])
            )
            request_nonce = self._nonce()
            permit_request = {
                "snapshot": manifest["snapshot"],
                "nonce": request_nonce,
                "issued_at": issued_at,
                "expires_at": expires_at,
                "model_tokens_requested": requested_tokens,
                "canonical_model_request": dict(request),
                "auth_mode": manifest["snapshot"]["auth_mode"],
                "budget_enforcement_mode": manifest["snapshot"][
                    "budget_enforcement_mode"
                ],
            }
            permit_auth_nonce = self._nonce()
            verify_auth_nonce = self._nonce()
            self._sequence += 1
            permit_exchange = _exchange(
                request=permit_request,
                operation="permit",
                sender_identity=manifest["peer_identity"],
                receiver_identity=manifest["server_identity"],
                key_id=manifest["peer_key_id"],
                nonce=permit_auth_nonce,
                sequence=self._sequence,
                private_key=self._peer_private_key,
            )
            self._sequence += 1
            verify_exchange = _exchange(
                request=permit_request,
                operation="verify",
                sender_identity=manifest["peer_identity"],
                receiver_identity=manifest["server_identity"],
                key_id=manifest["peer_key_id"],
                nonce=verify_auth_nonce,
                sequence=self._sequence,
                private_key=self._peer_private_key,
            )
            subject = {
                "schema_version": SUBJECT_SCHEMA_VERSION,
                "task_id": task_id,
                "hermes_run_id": self._hermes_run_id,
                "hcp_run_id": manifest["snapshot"]["run_id"],
                "bead_id": manifest["snapshot"]["bead_id"],
                "session_id": session_id,
                "profile_id": profile_id,
                "generation_id": manifest["generation_id"],
                "board_id": manifest["board_id"],
                "binding_digest": manifest["binding_digest"],
                "claim_lock_sha256": manifest["claim_lock_sha256"],
                "worker_pid": os.getpid(),
                "turn_id": turn_id,
                "api_request_id": api_request_id,
                "provider": provider,
                "model": model,
                "api_mode": api_mode,
                "endpoint_origin": endpoint_origin,
                "transport_identity_sha256": transport_identity_sha256,
                "transport_mode": transport_mode,
                "api_call_count": api_call_count,
            }
            subject_sha256 = _digest(subject)
            unsigned_wire = {
                "schema_version": WIRE_SCHEMA_VERSION,
                "operation": "permit_and_verify",
                "subject": subject,
                "subject_sha256": subject_sha256,
                "permit_exchange": permit_exchange,
                "verify_exchange": verify_exchange,
            }
            wire = {
                **unsigned_wire,
                "attestation_signature": _sign(self._peer_private_key, unsigned_wire),
            }
            response = self._transport.exchange(wire)
            return self._validate_response(
                response,
                permit_request=permit_request,
                request_sha256=request_sha256,
                subject_sha256=subject_sha256,
                permit_auth_nonce=permit_auth_nonce,
                verify_auth_nonce=verify_auth_nonce,
            )

    def _validate_response(
        self,
        response: Mapping[str, object],
        *,
        permit_request: Mapping[str, object],
        request_sha256: str,
        subject_sha256: str,
        permit_auth_nonce: str,
        verify_auth_nonce: str,
    ) -> ProviderRequestAuthorization:
        manifest = self._manifest
        if (
            not isinstance(response, Mapping)
            or set(response) != _RESULT_KEYS
            or response.get("schema_version") != RESULT_SCHEMA_VERSION
            or response.get("operation") != "permit_and_verify_result"
            or response.get("subject_sha256") != subject_sha256
            or response.get("permit_exchange_nonce") != permit_auth_nonce
            or response.get("verify_exchange_nonce") != verify_auth_nonce
            or response.get("server_identity") != manifest["server_identity"]
            or response.get("server_key_id") != manifest["server_key_id"]
            or _NONCE.fullmatch(str(response.get("response_nonce"))) is None
        ):
            raise ProviderRequestBlocked("HCP_PERMIT_RESPONSE_INVALID")
        response_nonce = str(response["response_nonce"])
        if response_nonce in self._used_nonces:
            raise ProviderRequestBlocked("HCP_PERMIT_RESPONSE_REPLAY")
        unsigned_response = dict(response)
        signature = unsigned_response.pop("signature")
        _verify(self._server_public_key, unsigned_response, signature)
        self._used_nonces.add(response_nonce)
        response_time = _timestamp(response.get("issued_at"))
        now = self._clock()
        if now.tzinfo != timezone.utc or response_time > now:
            raise ProviderRequestBlocked("HCP_PERMIT_RESPONSE_INVALID")

        outcome = response.get("outcome")
        if outcome != "PERMITTED":
            error_code = response.get("error_code")
            if type(error_code) is not str or _ERROR_CODE.fullmatch(error_code) is None:
                raise ProviderRequestBlocked("HCP_PERMIT_DENIED")
            raise ProviderRequestBlocked(error_code)
        if response.get("error_code") is not None:
            raise ProviderRequestBlocked("HCP_PERMIT_RESPONSE_INVALID")
        verification = response.get("verification")
        if (
            not isinstance(verification, Mapping)
            or set(verification) != {"valid", "error_code"}
            or verification.get("valid") is not True
            or verification.get("error_code") is not None
        ):
            raise ProviderRequestBlocked("HCP_PERMIT_VERIFICATION_FAILED")
        permit = response.get("permit")
        if not isinstance(permit, Mapping) or set(permit) != _PERMIT_KEYS:
            raise ProviderRequestBlocked("HCP_PERMIT_RESPONSE_INVALID")
        unsigned_permit = dict(permit)
        permit_signature = unsigned_permit.pop("signature")
        _verify(self._server_public_key, unsigned_permit, permit_signature)
        permit_id = _text(permit.get("permit_id"))
        issued = _timestamp(permit.get("issued_at"))
        expires = _timestamp(permit.get("expires_at"))
        now = self._clock()
        if (
            permit.get("schema_version") != PERMIT_SCHEMA_VERSION
            or permit.get("request_digest") != _digest(permit_request)
            or permit.get("snapshot") != permit_request["snapshot"]
            or permit.get("auth_mode") != permit_request["auth_mode"]
            or permit.get("budget_enforcement_mode")
            != permit_request["budget_enforcement_mode"]
            or permit.get("nonce") != permit_request["nonce"]
            or permit.get("model_token_limit")
            != permit_request["model_tokens_requested"]
            or permit.get("retry_budget_charge") != 0
            or permit.get("canonical_model_request_digest") != request_sha256
            or permit.get("signing_identity") != manifest["server_identity"]
            or permit.get("key_id") != manifest["server_key_id"]
            or issued > now
            or expires <= now
            or expires <= issued
            or (expires - issued).total_seconds() > manifest["permit_ttl_seconds"]
        ):
            raise ProviderRequestBlocked("HCP_PERMIT_RESPONSE_MISMATCH")
        if permit_id in self._used_permit_ids:
            raise ProviderRequestBlocked("HCP_PERMIT_REPLAY")
        self._used_permit_ids.add(permit_id)
        remaining_seconds = (expires - now).total_seconds()
        return ProviderRequestAuthorization(
            authorization_id=permit_id,
            request_sha256=request_sha256,
            subject_sha256=subject_sha256,
            expires_at_monotonic=time.monotonic() + remaining_seconds,
        )


def _environment_identity(profile_name: str) -> dict[str, str]:
    names = {
        "task_id": "HERMES_KANBAN_TASK",
        "hermes_run_id": "HERMES_KANBAN_RUN_ID",
        "board_id": "HERMES_KANBAN_BOARD",
        "profile_id": "HERMES_PROFILE",
    }
    identity = {key: os.environ.get(name, "").strip() for key, name in names.items()}
    if (
        any(not value for value in identity.values())
        or not identity["hermes_run_id"].isdecimal()
        or int(identity["hermes_run_id"]) <= 0
        or identity["profile_id"] != profile_name
    ):
        raise ProviderRequestBlocked("HCP_PERMIT_IDENTITY_UNAVAILABLE")
    claim_lock = os.environ.get("HERMES_KANBAN_CLAIM_LOCK", "")
    if not claim_lock:
        raise ProviderRequestBlocked("HCP_PERMIT_IDENTITY_UNAVAILABLE")
    identity["claim_lock_sha256"] = (
        "sha256:" + hashlib.sha256(claim_lock.encode("utf-8")).hexdigest()
    )
    return identity


def register(ctx) -> None:
    manifest_fd = os.environ.get("HCP_PRE_MODEL_PERMIT_MANIFEST_FD", "").strip()
    if not manifest_fd:
        # The bundled plugin is inert for ordinary Hermes processes. A process
        # marked REQUIRED remains fail-closed because PluginManager records the
        # requirement before discovery and no guard becomes available.
        return
    socket_path = os.environ.pop("HCP_PRE_MODEL_PERMIT_SOCKET", "").strip()
    try:
        manifest = _validate_manifest(
            _strict_json(
                _read_protected_fd(
                    "HCP_PRE_MODEL_PERMIT_MANIFEST_FD",
                    limit=_MAX_MANIFEST_BYTES,
                ),
                error_code="HCP_PERMIT_MANIFEST_INVALID",
            )
        )
        private_bytes = _read_protected_fd(
            "HCP_PRE_MODEL_PERMIT_PEER_PRIVATE_KEY_FD", limit=32
        )
        public_bytes = _read_protected_fd(
            "HCP_PRE_MODEL_PERMIT_SERVER_PUBLIC_KEY_FD", limit=32
        )
        if len(private_bytes) != 32 or len(public_bytes) != 32:
            raise ProviderRequestBlocked("HCP_PERMIT_KEY_INVALID")
        environment = _environment_identity(ctx.profile_name)
        if any(
            manifest[key] != environment[key]
            for key in ("task_id", "board_id", "profile_id", "claim_lock_sha256")
        ):
            raise ProviderRequestBlocked("HCP_PERMIT_IDENTITY_MISMATCH")
        guard = HCPPermitGuard(
            transport=UnixSocketPermitTransport(socket_path),
            manifest=manifest,
            peer_private_key=Ed25519PrivateKey.from_private_bytes(private_bytes),
            server_public_key=Ed25519PublicKey.from_public_bytes(public_bytes),
            hermes_run_id=environment["hermes_run_id"],
        )
    except (ProviderRequestBlocked, ValueError) as exc:
        error_code = getattr(exc, "error_code", "HCP_PERMIT_KEY_INVALID")
        raise RuntimeError(error_code) from exc
    finally:
        _close_remaining_worker_inputs()
        for name in (
            "HCP_PRE_MODEL_PERMIT_REQUIRED",
            "HCP_PRE_MODEL_PERMIT_ROOT_FD",
            "HCP_PRE_MODEL_PERMIT_MANIFEST_FD",
            "HCP_PRE_MODEL_PERMIT_PEER_PRIVATE_KEY_FD",
            "HCP_PRE_MODEL_PERMIT_SERVER_PUBLIC_KEY_FD",
        ):
            os.environ.pop(name, None)
    ctx.register_provider_request_guard(guard)


__all__ = [
    "HCPPermitGuard",
    "MANIFEST_SCHEMA_VERSION",
    "PERMIT_EXCHANGE_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "SUBJECT_SCHEMA_VERSION",
    "WIRE_SCHEMA_VERSION",
    "register",
]
