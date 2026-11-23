"""Launch warnings for paths TOP admits but no gate runs: nothing is known to break, nothing has verified them.

This module owns the launch-detectable list; the TOP README's "Not yet verified" section lists what only a run shows.
"""

import logging

from miles.utils.hf_utils.config import load_hf_config

logger = logging.getLogger(__name__)

# The widest expert parallelism a gate runs, on one node (Qwen3.5 MoE, GLM and DeepSeek-V4 train at EP4).
_GATED_EP = 4


def unverified_paths(args) -> list[str]:
    """What this launch uses that no TOP gate runs."""
    paths = []
    if args.recompute_granularity != "full":
        paths.append("training without full activation recompute (the gates pass --recompute-granularity full)")
    if not args.use_rollout_logprobs:
        paths.append("--use-rollout-logprobs off: the actor's forward-only pass runs under TOP")
    if args.kl_coef != 0 or args.use_kl_loss or args.keep_old_actor:
        paths.append("a reference or old-actor model (KL, --keep-old-actor): its forward-only pass runs under TOP")
    if args.colocate or args.offload_train:
        paths.append(
            "--colocate / --offload-train: TOP state built at model construction (rotary cache, published server "
            "args, DeepEP buffers) must survive sleep and wake-up"
        )
    world = args.actor_num_nodes * args.actor_num_gpus_per_node
    model_parallel = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    if not args.num_experts and world // max(model_parallel, 1) > 1:
        paths.append("a dense trainer at DP > 1")
    ep = args.expert_model_parallel_size or 1
    if ep > _GATED_EP or ep > args.actor_num_gpus_per_node:
        paths.append(f"trainer EP{ep}: the gates run EP up to {_GATED_EP} within one node")
    if args.use_rollout_routing_replay:
        paths.append(
            "rollout routing replay (--use-rollout-routing-replay): fresh tokens replay their own routes and stay "
            "exact; stale tokens train with the latest version's routes"
        )
    if args.prefill_num_servers is not None:
        paths.append("prefill/decode disaggregation")
    if args.sglang_disable_cuda_graph:
        paths.append("engines without CUDA graphs")
    if (
        args.sglang_speculative_algorithm is not None
        and not args.sglang_disable_radix_cache
        and _model_type(args) == "glm5_next"
    ):
        paths.append(
            "GLM-5.3-Flash radix reuse under speculation (a track crossed in a verify takes the deferred chunk "
            "close; a kernel test covers it, no gate)"
        )
    if args.sglang_page_size not in (None, 1) and _model_type(args) == "qwen3":
        paths.append("Qwen3 dense engines with --sglang-page-size > 1 (the paged FlashAttention path)")
    return paths


def warn_unverified_paths(args) -> None:
    if not args.true_on_policy_mode:
        return
    if args.fully_async or args.partial_rollout or args.max_weight_staleness is not None:
        logger.warning(
            "[top] fully-async / partial rollout / weight staleness: no TOP gate runs it, and a stale sample comes "
            "from older weights than the trainer scores it at, so its trainer and rollout log-probs are not expected "
            "to agree"
        )
    paths = unverified_paths(args)
    if paths:
        logger.warning(
            "[top] this launch uses paths that no TOP gate runs; exact log-probs are expected but not yet verified "
            "on them (README, 'Not yet verified'):\n%s",
            "\n".join(f"  - {path}" for path in paths),
        )


def _model_type(args) -> str | None:
    return load_hf_config(args.hf_checkpoint).model_type if args.hf_checkpoint else None
