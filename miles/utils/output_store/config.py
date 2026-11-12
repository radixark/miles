"""The ``--use-output-store`` switch: its launch checks and the run's store clients.

``RolloutRelatedConfig`` (``miles/utils/args/configs/rollout.py``) declares its flags, next to the object store's.

With the switch, training requests that ask for routed experts, indexer top-k or the sampling mask
also ask SGLang to put those arrays in the run's Mooncake (``radix_transfer_plugins``) and return a
ref. Every client Miles creates for it (SGLang engines, train workers, the rollout executor, the
session server) takes its config from ``output_store_backend_config``, which derives the connection
from Miles' own Mooncake config: one master for Miles' object store and the output store. At the end
of the run, ``sweep_output_store`` removes what is left under the run's key prefix.
"""

from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sglang.srt.server_args import ServerArgs

from miles.utils.environ import use_legacy_rollout_v1
from miles.utils.object_store_config import _parse_size, compute_mooncake_store_config
from miles.utils.rollout_topk_logprobs import requests_support_logprobs
from miles.utils.workers.argv_utils import _record_field_names
from miles.utils.workers.types import DeployComponent

if TYPE_CHECKING:
    from radix_transfer_plugins import OutputStore

    from miles.backends.megatron_utils.megatron_config import MegatronConfig
    from miles.backends.sglang_utils.sglang_config import SglangConfig

# SGLang's own fields for the store: Miles sets them on every engine it launches, so it exposes no
# --sglang-* flag or per-group override for them.
SGLANG_OUTPUT_STORE_FIELDS = ("output_store_backend", "output_store_backend_extra_config")

_BACKEND = "mooncake"
_KEY_PREFIX_BASE = "miles-output-store"


def check_output_store_args(args: argparse.Namespace, *, sglang: SglangConfig, megatron: MegatronConfig) -> None:
    """Reject, all at once, launch settings the output store does not serve."""
    if not args.use_output_store:
        return

    buffer_bytes = _parse_size(args.output_store_local_buffer_size)
    indexer_topk_bytes = _max_indexer_topk_bytes(args)
    # --save-debug-train-data stays allowed: the trainer dumps after its store columns are materialized.
    debug_data_flags = {
        "--save-debug-rollout-data (also set by --dump-details)": args.save_debug_rollout_data,
        "--load-debug-rollout-data": args.load_debug_rollout_data,
        "--ci-inject-rollout-data-path": args.ci_inject_rollout_data_path,
    }
    violations = [
        reason
        for violated, reason in (
            (
                not (
                    args.use_rollout_routing_replay
                    or args.use_rollout_indexer_replay
                    or args.use_rollout_sampling_support_replay
                    or requests_support_logprobs(args)
                ),
                "no request would use it: pass --use-rollout-routing-replay, --use-rollout-indexer-replay, "
                "--use-rollout-sampling-support-replay, or --rollout-top-logprobs-num with --rollout-top-p/--rollout-top-k",
            ),
            (
                args.object_store_backend != _BACKEND,
                f"--object-store-backend must be {_BACKEND}: the output store lives in the run's Mooncake, whose "
                f"master Miles starts only for that backend (got {args.object_store_backend})",
            ),
            (
                not _sglang_has_output_store(),
                "the installed SGLang has no output store "
                f"(ServerArgs lacks {' or '.join(SGLANG_OUTPUT_STORE_FIELDS)}), "
                "and Miles drops server args SGLang does not know",
            ),
            (
                use_legacy_rollout_v1(),
                "the legacy rollout (MILES_USE_LEGACY_ROLLOUT_V1) never asks SGLang for the output store",
            ),
            (
                args.rollout_external,
                "external rollout engines are not supported: Miles configures the store of the engines it launches",
            ),
            (
                sglang.has_pd_disaggregation or args.rollout_external_router_pd,
                "PD disaggregation is not supported: SGLang rejects the output store there",
            ),
            (
                args.custom_convert_samples_to_train_data_path is not None,
                "a custom convert function would bypass the output store plan column",
            ),
            (
                args.custom_async_data_buffer_path is not None,
                "a custom DataBuffer (--custom-async-data-buffer-path) may drop or restore groups without "
                "recording their output store refs",
            ),
            (
                megatron.is_multi_policy,
                "multi-policy training is not supported: stored objects are released per rollout id",
            ),
            (
                DeployComponent(args.deploy_component) is not DeployComponent.ALL,
                f"only --deploy-component {DeployComponent.ALL.value} is supported (got {args.deploy_component})",
            ),
            (args.debug_rollout_only, "--debug-rollout-only has no train worker to hold the stored objects"),
            (
                bool(args.multimodal_keys),
                "--multimodal-keys is not supported: a media prompt's routed experts have one row per expanded "
                "media token, not per prompt token",
            ),
            (
                indexer_topk_bytes is not None and indexer_topk_bytes > buffer_bytes,
                f"--output-store-local-buffer-size {args.output_store_local_buffer_size} cannot hold the "
                f"{indexer_topk_bytes} bytes of indexer top-k of the longest sequence known at launch from "
                "--max-seq-len or the --rollout-max-*-len flags",
            ),
            *(
                (value is not None, f"{flag} is not supported: saved output store refs do not outlive their run")
                for flag, value in debug_data_flags.items()
            ),
        )
        if violated
    ]
    if violations:
        raise ValueError("--use-output-store: " + "; ".join(violations))


def output_store_backend_config(
    args: argparse.Namespace, *, local_hostname: str | None = None, puts: bool = False
) -> dict[str, Any]:
    """The radix_transfer_plugins ``MooncakeConfig`` of a client of this run's output store.

    The connection comes from Miles' own Mooncake config. ``local_hostname`` and ``device_name`` belong
    to the client's node, which may not be the node building the config: an engine passes its host, and
    None leaves either to ``connect_output_store`` in the process that connects. Only a client that
    ``puts`` (an engine) registers a local buffer: readers read straight into their own memory.
    """
    init_kwargs = args.mooncake_store_init_kwargs or {}
    mooncake = compute_mooncake_store_config(init_kwargs, contribute_segment=False)
    return {
        "master_server_address": mooncake["master_server_addr"],
        "local_hostname": local_hostname,
        "local_buffer_size": args.output_store_local_buffer_size if puts else 0,
        # Writers only: one prefix per launch, so the run-end sweep finds exactly this run's objects.
        "key_prefix": _run_key_prefix(args),
        "protocol": mooncake["protocol"],
        "metadata_server": mooncake["metadata_server"],
        # MOONCAKE_DEVICE may differ by node, so only an init kwarg names the device for every client.
        "device_name": init_kwargs.get("device_name") or None,
        "replica_num": args.mooncake_replica_num,
    }


def sglang_output_store_server_args(args: argparse.Namespace, *, local_hostname: str) -> dict[str, str]:
    """The ``ServerArgs`` fields that make an engine write to the run's output store."""
    config = output_store_backend_config(args, local_hostname=local_hostname, puts=True)
    if config["device_name"] is None:
        # SGLang reads no MOONCAKE_* variable: without an init kwarg, the engine names Mooncake no device,
        # as Miles' own client does without MOONCAKE_DEVICE.
        del config["device_name"]
    return {"output_store_backend": _BACKEND, "output_store_backend_extra_config": json.dumps(config)}


def sweep_output_store(args: argparse.Namespace) -> int:
    """Remove every object of this run still in the store; returns how many keys it removed.

    Only for the end of the run: ``force`` removes keys under a read's lease too, which no reader holds by
    then. Each field of an object is its own key.
    """
    # mooncake ships with SGLang and the Mooncake object store backend; it is not a Miles dependency.
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    # A remove-only client: no segment, no local buffer.
    config = compute_mooncake_store_config(args.mooncake_store_init_kwargs or {}, contribute_segment=False)
    if error := store.setup({**config, "local_buffer_size": 0}):
        raise RuntimeError(f"Mooncake store setup failed: {error}")
    try:
        # run_uuid is lowercase hex, so the prefix holds no regex metacharacter.
        removed = store.remove_by_regex(f"^{_run_key_prefix(args)}/", True)
    finally:
        store.close()
    if removed < 0:
        raise RuntimeError(f"Mooncake remove_by_regex failed: {removed}")
    return removed


def connect_output_store(config: dict[str, Any]) -> OutputStore:
    """A client of the output store ``config`` names, for this process."""
    # radix_transfer_plugins ships with SGLang and is not a Miles dependency.
    from radix_transfer_plugins import OutputStore, create_backend

    # This node's address and device, resolved as Miles resolves its own Mooncake client's
    # (MOONCAKE_LOCAL_HOSTNAME or the node IP, and MOONCAKE_DEVICE).
    local = compute_mooncake_store_config({}, contribute_segment=False)
    node_values = {"local_hostname": local["local_hostname"], "device_name": local["rdma_devices"]}
    config = {**config, **{key: value for key, value in node_values.items() if config[key] is None}}

    return OutputStore(backend=create_backend(_BACKEND, json.dumps(config)))


def output_store_reader(args: argparse.Namespace) -> Callable | None:
    """This train worker's ``read`` for ``materialize_store_columns``, or None without the switch."""
    if not args.use_output_store:
        return None
    from radix_transfer_plugins.stashes.token_output import TokenOutputStash

    store = connect_output_store(output_store_backend_config(args))
    return functools.partial(store.read_many, stash_type=TokenOutputStash)


def _max_indexer_topk_bytes(args: argparse.Namespace) -> int | None:
    """Bytes of the indexer top-k of the longest sequence known at launch, or None if unknown.

    /generate caps every request at --max-seq-len. Under the session server the agent decides when to stop; one
    that stops once its prompt fills --max-seq-len sends at most a prompt one token short of it plus a full
    --rollout-max-response-len response, or a prompt of --max-seq-len when that is unset; an agent that ignores the
    limit can send more, which no launch check sees.
    Without --max-seq-len, a response cut at --rollout-max-response-len after a prompt of
    --rollout-max-prompt-len tokens, or of one when that is unset, is the longest sequence known at launch.
    Only this field is sized at launch: on DeepSeek-V3.2 it takes about 0.5 MB per token, routed experts take
    about 2 KB, and a sampling mask row holds the sampled support, which only sampling knows.
    """
    if not args.use_rollout_indexer_replay:
        return None
    streams, width = args.rollout_indexer_topk_num_streams, args.rollout_indexer_topk_width
    if not (streams and width):
        return None
    if args.max_seq_len is not None:
        num_tokens = args.max_seq_len
        if args.use_session_server and args.rollout_max_response_len is not None:
            num_tokens += args.rollout_max_response_len - 1
    elif args.rollout_max_response_len is not None:
        num_tokens = (args.rollout_max_prompt_len or 1) + args.rollout_max_response_len
    else:
        return None
    # One int32 row per forward step; no object outgrows the sequence it stores.
    return (num_tokens - 1) * streams * width * 4


def _run_key_prefix(args: argparse.Namespace) -> str:
    return f"{_KEY_PREFIX_BASE}/{args.run_uuid}"


def _sglang_has_output_store() -> bool:
    return set(SGLANG_OUTPUT_STORE_FIELDS) <= set(_record_field_names(ServerArgs))
