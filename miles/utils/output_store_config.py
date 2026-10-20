"""The run's output store: Miles' flags, launch checks, the SGLang engine config, and the trainer reader.

With ``--use-output-store``, SGLang writes R3, indexer top-k and the sampling mask of every training
response to an output store (``radix_transfer_plugins``) and returns a small ``output_store_ref``;
the trainer reads the objects back. Phase 1 has one backend: the run's own Mooncake, the master
Miles already uses for its object store. Every engine Miles launches writes to it and every train
worker reads from it.

The flags, the checks and every value derived from them live in this module only, so moving them
onto typed config later touches this module alone.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

from miles.utils.object_store_config import compute_mooncake_store_config, parse_size

if TYPE_CHECKING:
    from radix_transfer_plugins import OutputStore

# SGLang's own fields for the store. Miles sets them on every engine it launches, so it exposes no
# --sglang-* flag or per-group override for them.
SGLANG_OUTPUT_STORE_FIELDS = ("output_store_backend", "output_store_backend_extra_config")

_BACKEND = "mooncake"
_KEY_PREFIX_BASE = "miles-output-store"


def add_output_store_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--use-output-store",
        action="store_true",
        default=False,
        help=(
            "Return R3, indexer top-k and the sampling mask of training requests through the run's Mooncake "
            "output store instead of inline in the response; the trainer reads them, so this turns on "
            "--enable-delay-token-replay-assembly. Requires --object-store-backend mooncake and "
            "--pause-generation-mode in_place."
        ),
    )
    parser.add_argument(
        "--output-store-local-buffer-size",
        type=str,
        # Arbitrary until measured; bounds the largest single stored field one read can take.
        default="1gb",
        help=(
            "Local buffer of every output store client Miles creates (SGLang writers, trainer readers), "
            "in bytes or as a string such as '1gb'. A trainer reads in chunks that fit it, so it must hold "
            "the largest single stored field: one response's R3 or indexer top-k."
        ),
    )
    return parser


def validate_output_store_args(args: argparse.Namespace, *, multi_policy: bool, pd_disaggregation: bool) -> None:
    """Reject launch settings the output store cannot serve yet, all at once, and turn on delayed assembly."""
    if not args.use_output_store:
        return
    violations = [
        reason
        for violated, reason in (
            (
                args.object_store_backend != _BACKEND,
                f"--object-store-backend must be {_BACKEND}: the output store lives in the run's Mooncake, "
                f"whose master Miles starts only for that backend (got {args.object_store_backend})",
            ),
            (pd_disaggregation, "PD disaggregation is not supported: SGLang rejects the output store there"),
            (
                args.pause_generation_mode != "in_place",
                "--pause-generation-mode must be in_place: other modes are not adapted to the output store yet "
                f"(got {args.pause_generation_mode})",
            ),
            (
                "train" in args.ft_components,
                "train fault tolerance is not supported with the output store yet",
            ),
            (multi_policy, "multi-policy training is not supported with the output store yet"),
            (
                args.rollout_external,
                "external rollout engines are not supported: Miles configures the store of the engines it launches",
            ),
            (
                args.cluster_backend != "ray",
                f"only --cluster-backend ray is supported yet (got {args.cluster_backend})",
            ),
        )
        if violated
    ]
    if violations:
        raise ValueError("--use-output-store: " + "; ".join(violations))
    # --use-output-store wins over the default of --enable-delay-token-replay-assembly: the rollout executor
    # cannot read store objects, so only the train ranks can assemble the replay data. The flag is store_true,
    # so no launch asks for the opposite.
    args.enable_delay_token_replay_assembly = True


def compute_sglang_output_store_args(args: argparse.Namespace, *, local_hostname: str) -> dict[str, str]:
    """``ServerArgs`` fields that make an engine write to the run's output store."""
    config = _compute_backend_config(args, local_hostname=local_hostname)
    return {"output_store_backend": _BACKEND, "output_store_backend_extra_config": json.dumps(config)}


# ============================ trainer reader ============================

_READER: OutputStore | None = None


def init_output_store_reader(args: argparse.Namespace) -> None:
    """Connect this train worker's reader; a no-op without --use-output-store."""
    global _READER
    if not args.use_output_store:
        return
    assert _READER is None, "output store reader is already initialized"
    from radix_transfer_plugins import OutputStore, create_backend

    local_hostname = _resolve_mooncake_config(args)["local_hostname"]
    config = _compute_backend_config(args, local_hostname=local_hostname)
    _READER = OutputStore(backend=create_backend(_BACKEND, json.dumps(config)))


def get_output_store_reader() -> OutputStore:
    assert _READER is not None, (
        "the rollout data carries output store refs, but this process has no reader; "
        "was the run launched without --use-output-store?"
    )
    return _READER


def output_store_read_buffer_bytes(args: argparse.Namespace) -> int:
    return parse_size(args.output_store_local_buffer_size)


def _resolve_mooncake_config(args: argparse.Namespace, **overrides: Any) -> dict[str, Any]:
    init_kwargs = {**(args.mooncake_store_init_kwargs or {}), **overrides}
    return compute_mooncake_store_config(init_kwargs, contribute_segment=False)


def _compute_backend_config(args: argparse.Namespace, *, local_hostname: str) -> dict[str, Any]:
    """The package's ``MooncakeConfig``: connection fields come from Miles' own Mooncake config."""
    mooncake = _resolve_mooncake_config(args, local_hostname=local_hostname)
    return {
        "master_server_address": mooncake["master_server_addr"],
        "local_hostname": local_hostname,
        "local_buffer_size": args.output_store_local_buffer_size,
        # The run uuid survives a resume, so a resumed run keeps its keys under the same prefix.
        "key_prefix": f"{_KEY_PREFIX_BASE}/{args.run_uuid}",
        "protocol": mooncake["protocol"],
        "metadata_server": mooncake["metadata_server"],
        "device_name": mooncake["rdma_devices"],
        "replica_num": args.mooncake_replica_num,
    }
