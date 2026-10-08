"""Launch config of SGLang's output store for the engines Miles starts.

``--sglang-output-store-backend mooncake`` makes training responses carry an
``output_store_ref`` that the rollout executor reads through Miles' own Mooncake
object store, so every engine joins that cluster. Miles derives the connection,
key prefix and replica count from its object-store config and sets each engine's
own address. ``--sglang-output-store-backend-extra-config`` may add the remaining
keys; a key that contradicts a derived one is an error rather than an override,
because the rollout executor could not read what such an engine writes.
"""

import json
from typing import Any

from miles.rollout.generate_utils.output_store import output_store_enabled
from miles.utils.object_store import MOONCAKE_KEY_PREFIX, ObjectStoreBackend
from miles.utils.object_store_config import compute_mooncake_connection_config

# Arbitrary until measured. Miles' own 32 GiB default sizes trainer-side clients,
# and every SGLang tokenizer worker allocates one such buffer.
_DEFAULT_ENGINE_LOCAL_BUFFER_SIZE = "1gb"


def validate_output_store_args(args) -> None:
    if not output_store_enabled(args):
        return
    if ObjectStoreBackend(args.object_store_backend) != ObjectStoreBackend.MOONCAKE:
        raise ValueError(
            "--sglang-output-store-backend mooncake needs --object-store-backend mooncake: "
            "the rollout executor reads the engines' outputs through that store"
        )
    if args.prefill_num_servers is not None:
        raise ValueError("--sglang-output-store-backend mooncake does not support PD disaggregation yet")
    _checked_user_config(args, args.sglang_output_store_backend_extra_config)


def compute_engine_output_store_extra_config(args, *, user_config: str | None, local_hostname: str) -> str:
    """One engine's ``--output-store-backend-extra-config``."""
    return json.dumps(
        {
            "local_buffer_size": _DEFAULT_ENGINE_LOCAL_BUFFER_SIZE,
            **_checked_user_config(args, user_config),
            **_miles_owned_config(args),
            "local_hostname": local_hostname,
        }
    )


def _miles_owned_config(args) -> dict[str, Any]:
    return {
        **compute_mooncake_connection_config(args.mooncake_store_init_kwargs or {}),
        "key_prefix": MOONCAKE_KEY_PREFIX,
        "replica_num": args.mooncake_replica_num,
    }


def _checked_user_config(args, user_config: str | None) -> dict[str, Any]:
    config = json.loads(user_config) if user_config else {}
    if not isinstance(config, dict):
        raise ValueError(f"--sglang-output-store-backend-extra-config must be a JSON object, got {user_config!r}")
    if "local_hostname" in config:
        raise ValueError(
            "--sglang-output-store-backend-extra-config must not set local_hostname: "
            "Miles sets each engine's own address"
        )
    owned = _miles_owned_config(args)
    conflicts = {key: (config[key], owned[key]) for key in owned if key in config and config[key] != owned[key]}
    if conflicts:
        raise ValueError(
            "--sglang-output-store-backend-extra-config contradicts the Mooncake config "
            f"Miles reads the outputs with, as (given, derived): {conflicts}"
        )
    return config
