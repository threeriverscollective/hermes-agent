"""Private per-request Unix-socket transport for HCP permit exchange."""

from __future__ import annotations

import json
import math
import os
import socket
import stat
import struct
import time
from typing import Mapping

from hermes_cli.provider_request_guard import ProviderRequestBlocked


_FRAME_LIMIT = 2 * 1024 * 1024
_DEADLINE_SECONDS = 5.0


def canonical_json(value: object) -> bytes:
    """Encode the strict canonical JSON shared with HCP permit contracts."""

    def normalize(item: object) -> object:
        if item is None or type(item) in {bool, int, str}:
            if isinstance(item, str):
                item.encode("utf-8", errors="strict")
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("non-finite number")
            return item
        if isinstance(item, Mapping):
            if any(type(key) is not str for key in item):
                raise ValueError("non-string key")
            for key in item:
                normalize(key)
            return {key: normalize(item[key]) for key in sorted(item)}
        if isinstance(item, (tuple, list)):
            return [normalize(child) for child in item]
        raise ValueError("non-JSON value")

    try:
        encoded = json.dumps(
            normalize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProviderRequestBlocked("HCP_PERMIT_FRAME_INVALID") from exc
    if not 1 <= len(encoded) <= _FRAME_LIMIT:
        raise ProviderRequestBlocked("HCP_PERMIT_FRAME_INVALID")
    return encoded


def _strict_json(encoded: bytes) -> dict[str, object]:
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
        raise ProviderRequestBlocked("HCP_PERMIT_RESPONSE_INVALID") from exc
    if not isinstance(value, dict) or canonical_json(value) != encoded:
        raise ProviderRequestBlocked("HCP_PERMIT_RESPONSE_INVALID")
    return value


class UnixSocketPermitTransport:
    """Open a fresh authenticated-response channel for each provider attempt."""

    def __init__(
        self, socket_path: str, *, deadline_seconds: float = _DEADLINE_SECONDS
    ):
        if (
            type(socket_path) is not str
            or not os.path.isabs(socket_path)
            or not socket_path
            or type(deadline_seconds) not in {int, float}
            or isinstance(deadline_seconds, bool)
            or not 0 < float(deadline_seconds) <= 30
        ):
            raise ProviderRequestBlocked("HCP_PERMIT_CHANNEL_INVALID")
        self._socket_path = socket_path
        self._deadline_seconds = float(deadline_seconds)

    def exchange(self, request: Mapping[str, object]) -> dict[str, object]:
        encoded = canonical_json(request)
        try:
            before = os.lstat(self._socket_path)
        except OSError as exc:
            raise ProviderRequestBlocked("HCP_PERMIT_CHANNEL_UNAVAILABLE") from exc
        if (
            not stat.S_ISSOCK(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise ProviderRequestBlocked("HCP_PERMIT_CHANNEL_INVALID")

        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            channel.set_inheritable(False)
            channel.settimeout(self._deadline_seconds)
            channel.connect(self._socket_path)
            after = os.lstat(self._socket_path)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise ProviderRequestBlocked("HCP_PERMIT_CHANNEL_CHANGED")
            deadline = time.monotonic() + self._deadline_seconds
            self._write_all(
                channel,
                struct.pack(">I", len(encoded)) + encoded,
                deadline,
            )
            length = struct.unpack(">I", self._read_exact(channel, 4, deadline))[0]
            if not 1 <= length <= _FRAME_LIMIT:
                raise ProviderRequestBlocked("HCP_PERMIT_RESPONSE_INVALID")
            return _strict_json(self._read_exact(channel, length, deadline))
        except ProviderRequestBlocked:
            raise
        except (OSError, socket.timeout) as exc:
            raise ProviderRequestBlocked("HCP_PERMIT_CHANNEL_FAILED") from exc
        finally:
            channel.close()

    @staticmethod
    def _write_all(channel: socket.socket, value: bytes, deadline: float) -> None:
        remaining = memoryview(value)
        while remaining:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise ProviderRequestBlocked("HCP_PERMIT_CHANNEL_TIMEOUT")
            channel.settimeout(timeout)
            try:
                written = channel.send(remaining)
            except (OSError, socket.timeout) as exc:
                raise ProviderRequestBlocked("HCP_PERMIT_CHANNEL_FAILED") from exc
            if written <= 0:
                raise ProviderRequestBlocked("HCP_PERMIT_CHANNEL_FAILED")
            remaining = remaining[written:]

    @staticmethod
    def _read_exact(channel: socket.socket, length: int, deadline: float) -> bytes:
        chunks = bytearray()
        while len(chunks) < length:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise ProviderRequestBlocked("HCP_PERMIT_CHANNEL_TIMEOUT")
            channel.settimeout(timeout)
            try:
                chunk = channel.recv(length - len(chunks))
            except (OSError, socket.timeout) as exc:
                raise ProviderRequestBlocked("HCP_PERMIT_CHANNEL_FAILED") from exc
            if not chunk:
                raise ProviderRequestBlocked("HCP_PERMIT_CHANNEL_FAILED")
            chunks.extend(chunk)
        return bytes(chunks)


__all__ = ["UnixSocketPermitTransport", "canonical_json"]
