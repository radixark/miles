import argparse
import json

from sglang.srt.server_args import ServerArgs
from miles.utils.environ import use_legacy_rollout_v1
from miles.utils.http_utils import wrap_ipv6
from miles.utils.object_store import ObjectStoreBackend
from miles.utils.workers.argv_utils import _record_field_names


# TODO: use all sglang router arguments with `--sglang-router` prefix
def add_sglang_router_arguments(parser):
    """
    Add arguments to the parser for the SGLang router.
    """
    parser.add_argument(
        "--sglang-router-ip",
        type=str,
        default=None,
        help="IP address of the SGLang router",
    )
    parser.add_argument(
        "--sglang-router-port",
        type=int,
        default=None,
        help="Port of the SGLang router",
    )
    parser.set_defaults(sglang_model_routers=None)
    parser.add_argument(
        "--sglang-router-policy",
        type=str,
        default=None,
        help="Routing policy for the SGLang router (e.g., 'consistent_hashing', 'round_robin')",
    )
    parser.add_argument(
        "--sglang-router-request-timeout-secs",
        type=int,
        default=14400,
        help="Timeout for requests to the SGLang router in seconds",
    )
    return parser


_SKIPPED_SERVER_ARGS = [
    "model_path",
    "config",
    "trust_remote_code",
    "random_seed",
    # memory
    "enable_memory_saver",
    # distributed
    # tp_size stays exposed: scripts pass --sglang-tp-size, but the value is
    # overridden from --rollout-num-gpus-per-engine in validate_args below.
    "port",
    "nnodes",
    "node_rank",
    "dist_init_addr",
    "gpu_id_step",
    "base_gpu_id",
    "nccl_port",
    "gated_launch_port",
    "skip_server_warmup",
    "enable_return_routed_experts",
    "enable_return_indexer_topk",
    # Set from --sglang-output-store-backend and --sglang-output-store-client-config, declared below. The
    # client config is only part of the extra config, so its flag must not share this field's name: the
    # engine copies every args.sglang_<field> through.
    "output_store_backend",
    "output_store_backend_extra_config",
]

# The engines' output-store keys that shape only each engine's own Mooncake client. Miles derives
# every other key, so all engines write where the readers read.
_OUTPUT_STORE_CLIENT_KEYS = ("local_buffer_size", "chunk_bytes", "namespace", "partition")

# tp_size comes from --eval-num-gpus-per-engine, which also places the engines.
_EVAL_SKIPPED_SERVER_ARGS = _SKIPPED_SERVER_ARGS + ["tp_size"]


def _add_prefixed_server_args(parser, *, flag_prefix: str, dest_prefix: str, skipped_args: list[str], inherit: bool):
    """Register ``ServerArgs.add_cli_args`` as ``--{flag_prefix}-*`` -> ``args.{dest_prefix}*``.

    ``inherit=True`` defaults to SUPPRESS, so an unset flag leaves no attribute to inherit over.
    """
    old_add_argument = parser.add_argument

    def new_add_argument_wrapper(*name_or_flags, **kwargs):
        """
        Add arguments to the parser, ensuring that the server arguments are prefixed and skippable.
        """
        # Determine the canonical name for skip check (e.g., "model_path")
        canonical_name_for_skip_check = None
        if "dest" in kwargs:
            canonical_name_for_skip_check = kwargs["dest"]
        else:
            for flag_name_candidate in name_or_flags:
                if isinstance(flag_name_candidate, str) and flag_name_candidate.startswith("--"):
                    # Derive from first long flag: --foo-bar -> foo_bar
                    stem = flag_name_candidate[2:]
                    canonical_name_for_skip_check = stem.replace("-", "_")
                    break
            # If no long flag and no dest, skip logic might not catch it unless short flags imply a dest.

        if canonical_name_for_skip_check and canonical_name_for_skip_check in skipped_args:
            return  # Skip this entire argument definition

        # If not skipped, proceed to prefix flags and dest
        new_name_or_flags_list = []
        for item_flag in name_or_flags:
            if isinstance(item_flag, str) and item_flag.startswith("-"):
                original_flag_stem = item_flag.lstrip("-")  # "foo-bar" from "--foo-bar", or "f" from "-f"
                prefixed_item = f"--{flag_prefix}-{original_flag_stem}"
                new_name_or_flags_list.append(prefixed_item)
            else:
                # Positional arguments or non-string items
                new_name_or_flags_list.append(item_flag)

        # Prepare kwargs for the actual add_argument call.
        # Make a copy to avoid modifying the original kwargs dict.
        final_kwargs = kwargs.copy()

        # If 'dest' is explicitly provided and is a string, prefix it.
        # This ensures the attribute on the args namespace becomes, e.g., args.sglang_dest_name.
        if "dest" in final_kwargs and isinstance(final_kwargs["dest"], str):
            original_dest = final_kwargs["dest"]
            # Avoid double prefixing if dest somehow already starts with the prefix
            if not original_dest.startswith(dest_prefix):
                final_kwargs["dest"] = f"{dest_prefix}{original_dest}"
        elif "dest" not in final_kwargs:
            # argparse derives dest from the first alias, so store parallel sizes under SGLang's short field names.
            for item_flag in name_or_flags:
                if not isinstance(item_flag, str) or not item_flag.startswith("--"):
                    continue
                canonical_dest = item_flag[2:].replace("-", "_")
                if canonical_dest in ("tp_size", "dp_size", "pp_size", "ep_size"):
                    final_kwargs["dest"] = f"{dest_prefix}{canonical_dest}"
                    break

        if inherit:
            final_kwargs["default"] = argparse.SUPPRESS
            if final_kwargs.get("action") in ("store_true", "store_false"):
                # store_true can only turn a field on; an override must also turn one off.
                final_kwargs["action"] = argparse.BooleanOptionalAction
                final_kwargs.pop("const", None)
                final_kwargs.pop("nargs", None)

        old_add_argument(*new_name_or_flags_list, **final_kwargs)

    parser.add_argument = new_add_argument_wrapper
    ServerArgs.add_cli_args(parser)
    parser.add_argument = old_add_argument


def collect_eval_sglang_overrides(args) -> dict:
    """``ServerArgs`` fields set via ``--eval-sglang-*``; absent means inherit ``--sglang-*``."""
    return {
        key.removeprefix("eval_sglang_"): value for key, value in vars(args).items() if key.startswith("eval_sglang_")
    }


def add_sglang_arguments(parser):
    """
    Add arguments to the parser for the SGLang server.
    """
    parser = add_sglang_router_arguments(parser)
    parser.add_argument("--sglang-server-concurrency", type=int, default=512)
    parser.add_argument(
        "--sglang-output-store-backend",
        type=str,
        choices=("none", "mooncake"),
        default="none",
        help=(
            "Return training requests' replay outputs (routed experts, indexer top-k, sampling mask) through "
            "SGLang's output store instead of inline. 'mooncake' makes every rollout engine a client of Miles' "
            "Mooncake object store; Miles sets the engines' --output-store-backend and its extra config."
        ),
    )
    parser.add_argument(
        "--sglang-output-store-client-config",
        type=json.loads,
        default=None,
        help=(
            "JSON settings of each engine's own output-store client, merged into the extra config Miles derives. "
            f"Mooncake accepts {', '.join(_OUTPUT_STORE_CLIENT_KEYS)}; local_buffer_size, the host buffer every put "
            "is staged through (e.g. '4gb'), is required. Miles derives the connection, key prefix, replica "
            "count and local hostname."
        ),
    )

    _add_prefixed_server_args(
        parser, flag_prefix="sglang", dest_prefix="sglang_", skipped_args=_SKIPPED_SERVER_ARGS, inherit=False
    )
    _add_prefixed_server_args(
        parser,
        flag_prefix="eval-sglang",
        dest_prefix="eval_sglang_",
        skipped_args=_EVAL_SKIPPED_SERVER_ARGS,
        inherit=True,
    )

    parser.add_argument(
        "--sglang-config",
        type=str,
        default=None,
        help=(
            "Path to a YAML config for SGLang engine deployment. "
            "Defines server_groups with worker_type (regular/prefill/decode/placeholder), "
            "num_gpus per group, and optional per-group 'overrides' dict of "
            "ServerArgs field names that override the base --sglang-* CLI args. "
            "Placeholder groups reserve GPU slots without creating engines. "
            "A 'name: eval' model is filled in from the --eval-* args (model_path, "
            "num_gpus_per_engine, --eval-sglang-* overrides) wherever the YAML leaves them unset. "
            "Mutually exclusive with --prefill-num-servers."
        ),
    )

    return parser


def validate_args(args):
    args.sglang_tp_size = args.rollout_num_gpus_per_engine

    if getattr(args, "loss_type", None) == "score_centering":
        # Score centering forces the base LM-head setting on, overriding the parsed
        # value to avoid lower-precision logits rounding before rollout probabilities.
        # This trades serving compute/memory for precision; per-engine overrides still win.
        args.sglang_enable_fp32_lm_head = True

    if args.true_on_policy_mode:
        # True on-policy mode overrides the parsed setting to keep inference deterministic.
        args.sglang_enable_deterministic_inference = True

    if getattr(args, "recompute_logprobs_via_prefill", False):
        # Prefill logprob recomputation overrides both parsed settings for deterministic scoring.
        args.sglang_enable_prefill_only_deterministic_inference = True
        args.sglang_enable_deterministic_inference = True

    if args.sglang_dp_size > 1:
        assert args.sglang_enable_dp_attention

    if args.sglang_enable_dp_attention:
        args.router_dp_aware = True

    if args.sglang_router_policy is None and args.use_session_server:
        args.sglang_router_policy = "manual"
        if args.router_assignment_mode == "random":
            args.router_assignment_mode = "min_load"

    if getattr(args, "sglang_router_ip", None):
        args.sglang_router_ip = wrap_ipv6(args.sglang_router_ip)

    _validate_output_store_args(args)


def _validate_output_store_args(args) -> None:
    if args.sglang_output_store_backend == "none":
        return
    flag = f"--sglang-output-store-backend {args.sglang_output_store_backend}"
    assert "output_store_backend" in _record_field_names(
        ServerArgs
    ), f"{flag} needs an SGLang with --output-store-backend"
    assert ObjectStoreBackend(args.object_store_backend) == ObjectStoreBackend.MOONCAKE, (
        f"{flag} needs --object-store-backend mooncake: the rollout executor and the session server "
        "read the engines' outputs through that store"
    )
    assert (
        args.use_rollout_routing_replay or args.use_rollout_indexer_replay or args.use_sampling_support_replay
    ), f"{flag} only moves replay outputs; enable routing, indexer or sampling-support replay"
    assert args.prefill_num_servers is None, f"{flag} does not support PD disaggregation yet"
    assert not args.rollout_external, f"{flag} supports only the engines Miles launches"
    assert not use_legacy_rollout_v1(), f"{flag} needs the class-based rollout API; unset MILES_USE_LEGACY_ROLLOUT_V1"

    client_config = args.sglang_output_store_client_config or {}
    rejected = sorted(set(client_config) - set(_OUTPUT_STORE_CLIENT_KEYS))
    assert not rejected, (
        f"--sglang-output-store-client-config accepts only {', '.join(_OUTPUT_STORE_CLIENT_KEYS)}, "
        f"got {rejected}; Miles derives the connection, key prefix, replica count and local hostname"
    )
    assert "local_buffer_size" in client_config, (
        f'{flag} needs --sglang-output-store-client-config \'{{"local_buffer_size": "<size>"}}\': '
        "the host buffer each engine stages its puts through has no measured default"
    )
