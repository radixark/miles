from miles.utils.args.schema import A, Arg, BaseConfig


class AlgoConfig(BaseConfig):
    ref_load: A[
        str | None,
        Arg(
            help=(
                "The checkpoint for reference model. "
                "When --load is not set, this will be used as the initial checkpoint for training. "
            )
        ),
    ] = None
    ref_ckpt_step: A[
        int | None,
        Arg(help="The checkpoint step for reference model. "),
    ] = None
    load: A[str | None, Arg(reset=True)] = None
    save: A[str | None, Arg(reset=True)] = None
    save_interval: A[int | None, Arg(reset=True)] = None
    async_save: A[bool, Arg(reset=True)]
    save_hf: A[
        str | None,
        Arg(
            help=(
                "Path to save the model in HuggingFace format when using Megatron backend. "
                "The model will be saved to `save_hf.format(rollout_id)`. "
            )
        ),
    ] = None
    save_trigger_sentinel: A[
        str | None,
        Arg(
            help=(
                "Path to a sentinel file for externally-triggered checkpoint saving. If the file "
                "exists at an iteration's save point, a checkpoint is saved and the file is removed."
            )
        ),
    ] = None
    custom_megatron_post_save_hook_path: A[
        str | None,
        Arg(
            help=(
                "Path to a custom function invoked on rank 0 after every checkpoint save. "
                "Signature: def hook(args, rollout_id: int, checkpoint_dir: str, "
                "hf_checkpoint_dir: str | None) -> None."
            )
        ),
    ] = None
    seed: A[int, Arg(reset=True)] = 1234
    num_critic_only_steps: A[
        int,
        Arg(
            help=(
                "Number of initial rollout steps where only the critic trains (value-function warmup) "
                "while the actor stays frozen. Only takes effect when --advantage-estimator is ppo."
            )
        ),
    ] = 0
    critic_load: A[str | None, Arg(help="The checkpoint for critic model.")] = None
    critic_save: A[
        str | None,
        Arg(
            help=(
                "Where to save critic checkpoints. If not set, it defaults to the --save path with a "
                "'_critic' suffix appended, e.g. --save /ckpts/run1 saves the critic to /ckpts/run1_critic."
            )
        ),
    ] = None
    critic_lr: A[float | None, Arg(help="The lr for critic model")] = None
    critic_lr_warmup_iters: A[
        int,
        Arg(help="number of iterations to linearly warmup for critic model."),
    ] = 0

    eps_clip: A[float, Arg(help="PPO clip range")] = 0.2
    eps_clip_high: A[float | None, Arg(help="PPO clip upper range")] = None
    eps_clip_c: A[
        float | None,
        Arg(help="lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729"),
    ] = None
    value_clip: A[float, Arg(help="the clip for value loss")] = 0.2
    kl_coef: A[
        float,
        Arg(
            help="KL penalty coefficient for reward shaping. This is applied to the reward signal before advantage calculation."
        ),
    ] = 0.00
    loss_type: A[
        str,
        Arg(
            choices=["policy_loss", "sft_loss", "custom_loss"],
            help=(
                "Choose loss type, currently support ppo policy_loss or sft_loss, "
                "if custom_loss is set, we will use the function path from `--custom-loss-function-path`."
            ),
        ),
    ] = "policy_loss"
    custom_loss_function_path: A[
        str | None,
        Arg(
            help=(
                "Path to the custom loss function, if the loss_type is `custom_loss`, "
                "we will use this function to calculate the loss. "
            )
        ),
    ] = None
    kl_loss_type: A[
        str,
        Arg(
            choices=["k1", "k2", "k3", "low_var_kl"],
            help="Choose KL loss type: kl, k2, k3, low_var_kl",
        ),
    ] = "k1"
    advantage_estimator: A[
        str,
        Arg(
            choices=["grpo", "gspo", "reinforce_plus_plus", "reinforce_plus_plus_baseline", "ppo"],
            help=(
                "Advantage estimator to use. Note: on-policy distillation (OPD) is now orthogonal "
                "to the advantage estimator. Use --opd-kl-coef > 0 to enable OPD on top of any estimator."
            ),
        ),
    ] = "grpo"
    compute_advantages_and_returns: A[
        bool,
        Arg(
            cli_name="--disable-compute-advantages-and-returns",
            action="store_false",
            help=(
                "Whether to disable computing advantages and returns. "
                "If set, we will not compute the advantages and returns, "
                "This is useful for sft or custom loss function."
            ),
        ),
    ] = True
    use_kl_loss: A[bool, Arg(help="whether to use KL loss from GRPO")] = False
    kl_loss_coef: A[
        float,
        Arg(help="KL penalty coefficient for the loss function. This is added to the final PPO loss."),
    ] = 0.0
    use_unbiased_kl: A[
        bool,
        Arg(help="Whether to enable unbiased KL estimation."),
    ] = False
    ref_update_interval: A[
        int | None,
        Arg(help="Interval (in rollout steps) to update ref model from actor. If None, ref model is not updated."),
    ] = None
    entropy_coef: A[float, Arg(help="Entropy loss coef")] = 0.0
    gamma: A[float, Arg(help="PPO GAE gamma")] = 1.0
    lambd: A[float, Arg(help="PPO GAE lambd")] = 1.0
    normalize_advantages: A[bool, Arg()] = False
    grpo_std_normalization: A[
        bool,
        Arg(
            cli_name="--disable-grpo-std-normalization",
            action="store_false",
            help="from Dr.GRPO https://arxiv.org/pdf/2503.20783",
        ),
    ] = True
    rewards_normalization: A[
        bool,
        Arg(
            cli_name="--disable-rewards-normalization",
            action="store_false",
            help="Disable rewards normalization",
        ),
    ] = True
    use_rollout_entropy: A[
        bool,
        Arg(
            help=(
                "Whether to calculate the entropy when calculating the logprobs from actor and reference model. "
                "This is useful for doing special loss mask."
            )
        ),
    ] = False
    observe_training_entropy: A[
        bool,
        Arg(
            help=(
                "Compute training entropy as a logged metric even when --entropy-coef is 0. "
                "When the coefficient is 0, the observed entropy is detached and does not affect backward."
            )
        ),
    ] = False
    get_mismatch_metrics: A[
        bool,
        Arg(help="Whether to calculate the mismatch metrics."),
    ] = False
    reset_optimizer_states: A[
        bool,
        Arg(
            help=(
                "Whether to reset optimizer states after each rollout. "
                "If enabled, the optimizer's history will be cleared at the end of each rollout, which can sometimes help with training stability or fulfill specific experiment requirements."
            )
        ),
    ] = False
    use_rollout_logprobs: A[
        bool,
        Arg(
            help=(
                "Whether to use the rollout logprobs when calculating the importance sampling ratios. "
                "If not set, we will use the logprobs from the actor model."
            )
        ),
    ] = False
    skip_actor_forward_only: A[
        bool,
        Arg(
            help=(
                "Skip the standalone Megatron actor forward-only pass. With --use-rollout-logprobs, "
                "those log-probs remain the old-policy baseline; otherwise detached training log-probs "
                "are reused and the actor importance log-ratio is exactly 0. This requires a single "
                "optimizer step. The skipped pass's rollout/log_probs metric is not emitted."
            )
        ),
    ] = False
    # Off-Policy Correction using Importance Sampling: https://fengyao.notion.site/off-policy-rl
    use_tis: A[
        bool,
        Arg(help="Enable TIS from https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33."),
    ] = False
    tis_clip: A[
        float,
        Arg(help="Clipping threshold C for importance sampling ratios to control variance."),
    ] = 2.0
    tis_clip_low: A[
        float,
        Arg(help="Lower bound clipping threshold C for importance sampling ratios to control variance."),
    ] = 0
    custom_tis_function_path: A[
        str | None,
        Arg(
            help="Path to the custom TIS/RS function (e.g., examples/infra_features/train_infer_mismatch_helper/mis.py:compute_mis_weights_with_cp)."
        ),
    ] = None
    custom_pg_loss_reducer_function_path: A[
        str | None,
        Arg(
            help="Path to a custom reducer function for pg_loss only. When set, pg_loss will use this custom reducer while other metrics (pg_clipfrac, ppo_kl, entropy_loss, etc.) still use the default sum_of_sample_mean. (e.g., examples/experimental/DrGRPO/custom_reducer.py:get_pg_loss_reducer)."
        ),
    ] = None

    use_routing_replay: A[
        bool,
        Arg(help="The routing replay technique from https://arxiv.org/abs/2507.18071"),
    ] = False
    use_rollout_routing_replay: A[
        bool,
        Arg(
            help=(
                "The rollout routing replay technique from https://arxiv.org/abs/2510.11370 (R3): "
                "replay the rollout's MoE routing in training. MoE-only; the GLM-5 launchers pass it "
                "explicitly."
            )
        ),
    ] = False
    use_indexer_replay: A[
        bool,
        Arg(help="Replay indexer topk decisions for layers with indexers."),
    ] = False
    use_rollout_indexer_replay: A[
        bool,
        Arg(help="Replay indexer topk from rollout during training."),
    ] = False
    use_opsm: A[
        bool,
        Arg(help="Whether to enable Off-Policy Sequence Masking (OPSM)."),
    ] = False
    opsm_delta: A[
        float,
        Arg(help="The threshold for Off-Policy Sequence Masking (OPSM)."),
    ] = 1e-4
