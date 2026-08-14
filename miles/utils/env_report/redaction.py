import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

_SECRET_ENV_VAR_PATTERN = re.compile(
    r"(^|_)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|DATABASE_URL)$", re.IGNORECASE
)
_SGLANG_SECRET_ARG_BASE_NAMES = frozenset({"api_key", "admin_api_key", "ssl_keyfile_password"})
_SECRET_ARG_NAMES = frozenset(
    {
        "wandb_key",
        "router_api_key",
        "router_oracle_password",
        "router_control_plane_api_keys",
        *(f"{prefix}{name}" for prefix in ("sglang_", "eval_sglang_") for name in _SGLANG_SECRET_ARG_BASE_NAMES),
    }
)
_ENV_VAR_ARG_NAMES = frozenset({"train_env_vars"})
_REDACTED_PREFIX = "redacted-sha256:"
_REDACTED_HASH_CHARS = 16


def redact_env_vars(env_vars: dict[str, str]) -> dict[str, str]:
    return {
        name: _redact(value) if _SECRET_ENV_VAR_PATTERN.search(name) else value
        for name, value in sorted(env_vars.items())
    }


def redact_argv(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_value: Callable[[str], str] | None = None
    for item in argv:
        if redact_value is not None and not item.startswith("-"):
            redacted.append(redact_value(item))
            continue

        redact_value = None
        flag, separator, value = item.partition("=")
        if (rewrite := _ARGV_VALUE_REDACTORS.get(flag)) is None:
            redacted.append(item)
            continue

        redacted.append(f"{flag}={rewrite(value)}" if separator else item)
        redact_value = None if separator else rewrite

    return redacted


def redact_server_info(server_info: dict[str, Any]) -> dict[str, Any]:
    redacted = _redact_server_args(server_info)

    if isinstance(internal_states := redacted.get("internal_states"), list):
        redacted["internal_states"] = [_redact_internal_state(state) for state in internal_states]

    return redacted


def redact_arg(name: str, value: Any) -> Any:
    if name in _ENV_VAR_ARG_NAMES and isinstance(value, dict):
        return redact_env_vars({key: str(item) for key, item in value.items()})
    if name not in _SECRET_ARG_NAMES:
        return value
    return _redact_secret_value(value)


def _redact(value: str) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:_REDACTED_HASH_CHARS]
    return f"{_REDACTED_PREFIX}{digest}"


def _argv_value_redactors() -> dict[str, Callable[[str], str]]:
    def flag(name: str) -> str:
        return f"--{name.replace('_', '-')}"

    return {
        **{flag(name): _redact for name in _SECRET_ARG_NAMES},
        **{flag(name): _redact_env_var_json for name in _ENV_VAR_ARG_NAMES},
    }


def _redact_env_var_json(value: str) -> str:
    try:
        env_vars = json.loads(value)
    except json.JSONDecodeError:
        return _redact(value)
    if not isinstance(env_vars, dict):
        return _redact(value)
    return json.dumps(redact_env_vars({name: str(item) for name, item in env_vars.items()}))


def _redact_internal_state(state: Any) -> Any:
    if not isinstance(state, dict):
        return state

    redacted = _redact_server_args(state)
    if isinstance(env_vars := redacted.get("env_vars"), dict):
        redacted["env_vars"] = redact_env_vars({name: str(value) for name, value in env_vars.items()})
    return redacted


def _redact_server_args(values: dict[str, Any]) -> dict[str, Any]:
    return {
        name: _redact_secret_value(value) if name in _SGLANG_SECRET_ARG_BASE_NAMES else value
        for name, value in values.items()
    }


def _redact_secret_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, list):
        return [_redact(item) if isinstance(item, str) else item for item in value]
    return value


_ARGV_VALUE_REDACTORS = _argv_value_redactors()
