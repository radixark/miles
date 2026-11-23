"""Launch refusals for run features that change the TOP program and that no TOP gate runs."""

from sglang.srt.true_on_policy.contracts import TRUE_ON_POLICY_V1
from sglang.srt.true_on_policy.registry import Family
from sglang.srt.true_on_policy.sampling import validate_sampling_temperature

from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.true_on_policy.unverified import warn_unverified_paths
from miles.utils.lora.utils import is_lora_enabled, is_multi_lora_enabled

# The TOP router op whose serving router (TopQwenRouter) captures routes and whose trainer router replays them.
# Qwen3 MoE also lists it but serves through TopQwenMoe, which refuses capture; it is admitted at no layout.
_REPLAYING_ROUTER = "qwen_router"


def validate_true_on_policy_args(args) -> None:
    """Refuse at startup what would otherwise drift silently or fail hours into the run."""
    if not args.true_on_policy_mode:
        return
    if is_lora_enabled(args) or is_multi_lora_enabled(args):
        raise ValueError(
            "[top] LoRA adds the adapter output with SGLang's LoRA kernels in serving and Megatron's adapter "
            "modules in the trainer, so their log-probs differ; drop --lora-rank, --lora-adapter-path and "
            "--multi-lora-n-adapters"
        )
    if args.attention_dropout != 0:
        raise ValueError(
            f"[top] TOP attention runs without dropout, so --attention-dropout {args.attention_dropout} (Megatron's "
            "default is 0.1) would be ignored; pass --attention-dropout 0.0"
        )
    if args.entropy_coef != 0:
        raise ValueError(
            "[top] an entropy loss (--entropy-coef != 0) scores log-probs through a full-logit BF16 copy instead of "
            "the gated row-chunked TOP path; pass --entropy-coef 0"
        )
    if args.use_critic:
        raise ValueError(
            "[top] the PPO critic would be built from the TOP block spec and run the serving program, which no gate "
            "covers; use a critic-free estimator such as --advantage-estimator grpo"
        )
    # --use-rollout-routing-replay also sets use_routing_replay; only trainer-side recording is refused.
    if args.use_routing_replay and not args.use_rollout_routing_replay:
        raise ValueError(
            "[top] --use-routing-replay records the trainer's own routes, which TOP already computes exactly; use "
            "--use-rollout-routing-replay to replay the rollout's routes"
        )
    if args.rollout_top_logprobs_num:
        raise ValueError(
            "[top] --rollout-top-logprobs-num: the rollout top-k candidates are scored against the trainer only by "
            "the torch-native actor, which TOP does not run, so the --ci-test check would fail on missing scores "
            "and no gate covers the candidates; drop --rollout-top-logprobs-num"
        )
    if args.rollout_external:
        raise ValueError(
            "[top] external engines and custom engine providers are not launched by Miles, so nothing applies the "
            "contract to them; drop --rollout-external-engine-addrs and --custom-inference-engine-provider-path"
        )
    if args.multimodal_keys is not None and not args.multimodal_token_space_processing:
        raise ValueError(
            "[top] TOP trains samples with media on the engines' own pixels, placeholder placement and mRoPE "
            "positions, which the trainer rebuilds with SGLang's token-space processor; pass "
            "--multimodal-token-space-processing and --sglang-enable-token-space-processor"
        )
    if args.multimodal_keys is not None and "audio" in args.multimodal_keys:
        raise ValueError(
            "[top] TOP has a program for the vision tower only (images and video); drop the audio entry of "
            "--multimodal-keys"
        )
    _refuse_transformed_sampling(args)
    warn_unverified_paths(args)


def refuse_engines_off_the_contract(args, *, sglang: SglangConfig) -> None:
    """Every engine whose samples train the policy must be launched by Miles with the TOP server args.

    Runs on the resolved SGLang config, after the argument checks of ``validate_true_on_policy_args``."""
    if not args.true_on_policy_mode:
        return
    # ServerArgs fields Miles sets on every engine of a TOP run; an engine without them serves another program.
    contract = args.true_on_policy_contract or TRUE_ON_POLICY_V1
    wanted = {"true_on_policy_contract": contract, "enable_deterministic_inference": True}
    for model in sglang.models:
        if not model.update_weights:
            continue
        for group in model.server_groups:
            changed = sorted(key for key in wanted if group.overrides.get(key, wanted[key]) != wanted[key])
            if changed:
                raise ValueError(
                    f"[top] --sglang-config model {model.name!r} overrides {', '.join(changed)}, which TOP sets on "
                    "every engine of the trained policy; remove the override"
                )


def refuse_unreplayable_routing(args, *, family: Family) -> None:
    """Rollout routing replay needs a TOP router that captures routes in serving and replays them in training.

    Runs on the family ``admit_trainer`` resolved from the checkpoint."""
    if not args.true_on_policy_mode or not args.use_rollout_routing_replay:
        return
    if _REPLAYING_ROUTER not in family.ops:
        raise ValueError(
            f"[top] --use-rollout-routing-replay: only the Qwen softmax router (op {_REPLAYING_ROUTER}) captures and "
            f"replays rollout routes under TOP, and {family.name} does not route with it; drop the flag"
        )


def _refuse_transformed_sampling(args) -> None:
    """SGLang refuses these per request under the contract, so the first eval would fail mid-run instead."""
    _refuse_greedy("--rollout-temperature", args.rollout_temperature)
    for dataset in args.eval_datasets:
        source = f"eval dataset {dataset.name!r}"
        _refuse_greedy(source, dataset.temperature)
        transforms = [
            name
            for name, transformed in (
                ("top_p", dataset.top_p not in (None, 1.0)),
                ("top_k", dataset.top_k not in (None, -1)),
                ("min_new_tokens", dataset.min_new_tokens not in (None, 0)),
            )
            if transformed
        ]
        if transforms:
            raise ValueError(
                f"[top] {source} sets {', '.join(transforms)}; TOP samples the full, unmodified distribution, "
                "so drop them from --eval-top-p/--eval-top-k or the --eval-config dataset"
            )


def _refuse_greedy(source: str, temperature: float | None) -> None:
    if temperature is None:
        return
    try:
        validate_sampling_temperature(temperature)
    except ValueError as exc:
        raise ValueError(f"[top] {source}: {exc}") from None
