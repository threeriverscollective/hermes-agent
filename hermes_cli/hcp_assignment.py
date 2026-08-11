"""Hermes-owned, provider-free assignment for admitted HCP cards.

HCP supplies a closed allowlist and an authenticated profile-catalog root.
Hermes selects only when exactly one allowlisted profile exists.  This command
does not claim a task, launch a worker, invoke a model, or mutate a board.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any


HCP_CLI_COMMAND = "hcp-assignment"
HCP_AUXILIARY_TASK = "hcp_assignment_only"
NO_FIT = "NO_FIT"
_INPUT_KEYS = frozenset(
    {
        "schema_version",
        "card_id",
        "normalized_title",
        "capsule_summary",
        "allowed_profile_ids",
        "required_capability_classes",
        "repository_policy_id",
        "routing_policy_id",
    }
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class AssignmentRefusal(ValueError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


def _identifier(value: object) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise AssignmentRefusal("INPUT_SCHEMA")
    return value


def _text(value: object, maximum: int) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or "\x00" in value
    ):
        raise AssignmentRefusal("INPUT_SCHEMA")
    return value


def _sorted_identifiers(value: object, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or len(value) > 64 or (not value and not allow_empty):
        raise AssignmentRefusal("INPUT_SCHEMA")
    result = [_identifier(item) for item in value]
    if result != sorted(set(result)):
        raise AssignmentRefusal("INPUT_SCHEMA")
    return result


def _request(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _INPUT_KEYS:
        raise AssignmentRefusal("INPUT_SCHEMA")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise AssignmentRefusal("INPUT_SCHEMA")
    return {
        "schema_version": 1,
        "card_id": _identifier(value["card_id"]),
        "normalized_title": _text(value["normalized_title"], 256),
        "capsule_summary": _text(value["capsule_summary"], 4096),
        "allowed_profile_ids": _sorted_identifiers(
            value["allowed_profile_ids"], allow_empty=True
        ),
        "required_capability_classes": _sorted_identifiers(
            value["required_capability_classes"]
        ),
        "repository_policy_id": _identifier(value["repository_policy_id"]),
        "routing_policy_id": _identifier(value["routing_policy_id"]),
    }


def _profiles_root(value: object) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise AssignmentRefusal("PROFILE_CATALOG_INVALID")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise AssignmentRefusal("PROFILE_CATALOG_INVALID")
    current = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            current /= component
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode):
                raise AssignmentRefusal("PROFILE_CATALOG_INVALID")
        metadata = os.lstat(path)
    except AssignmentRefusal:
        raise
    except OSError as error:
        raise AssignmentRefusal("PROFILE_CATALOG_INVALID") from error
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise AssignmentRefusal("PROFILE_CATALOG_INVALID")
    return path


def _installed(root: Path, profile_id: str) -> bool:
    path = root / profile_id
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise AssignmentRefusal("PROFILE_CATALOG_INVALID") from error
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
    )


def route_assignment(
    value: object,
    *,
    profiles_root: str | Path,
    run_id: str,
    provider: str,
    model: str,
    effort: str,
    attempted_profile_id: str,
) -> dict[str, object]:
    """Select one exact installed profile without any provider or side effect."""

    request = _request(value)
    root = _profiles_root(str(profiles_root))
    _identifier(run_id)
    identity = {
        "provider": _identifier(provider),
        "model": _identifier(model),
        "effort": _identifier(effort),
        "attempted_profile_id": _identifier(attempted_profile_id),
    }
    if identity["attempted_profile_id"] != HCP_AUXILIARY_TASK:
        raise AssignmentRefusal("AUXILIARY_CONTEXT_MISMATCH")
    allowed = request["allowed_profile_ids"]
    assert isinstance(allowed, list)
    installed = [profile_id for profile_id in allowed if _installed(root, profile_id)]
    selection = installed[0] if len(allowed) == 1 and installed == allowed else NO_FIT
    encoded = json.dumps(
        request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "selection": selection,
        "usage": {
            **identity,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_tokens": 0,
            "total_tokens": 0,
            "cost_microunits": 0,
            "context_bytes": len(encoded),
            "tool_bytes": 0,
            "wall_time_ms": 0,
            "queue_time_ms": 0,
            "retry_count": 0,
            "termination_reason": "no_fit" if selection == NO_FIT else "completed",
            "broker_api_calls": 0,
            "quota_units": 0,
        },
    }


def _register_cli(parser: Any) -> None:
    parser.add_argument("--input-json", action="store_true", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--attempted-profile-id", required=True)
    parser.add_argument("--profiles-root", required=True)


def _strict_stdin() -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise AssignmentRefusal("INPUT_SCHEMA")
            result[key] = value
        return result

    try:
        return json.loads(
            sys.stdin.read(),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except AssignmentRefusal:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AssignmentRefusal("INPUT_SCHEMA") from error


def hcp_assignment_command(args: Any) -> int:
    try:
        response = route_assignment(
            _strict_stdin(),
            profiles_root=args.profiles_root,
            run_id=args.run_id,
            provider=args.provider,
            model=args.model,
            effort=args.effort,
            attempted_profile_id=args.attempted_profile_id,
        )
    except AssignmentRefusal as error:
        print(json.dumps({"error_code": error.error_code}, separators=(",", ":")))
        return 2
    print(json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def register(ctx: Any) -> None:
    ctx.register_cli_command(
        name=HCP_CLI_COMMAND,
        help="Select one installed profile from a closed HCP allowlist",
        setup_fn=_register_cli,
        handler_fn=hcp_assignment_command,
        description="Provider-free assignment-only HCP routing bridge.",
    )


__all__ = [
    "AssignmentRefusal",
    "HCP_AUXILIARY_TASK",
    "HCP_CLI_COMMAND",
    "NO_FIT",
    "hcp_assignment_command",
    "register",
    "route_assignment",
]
