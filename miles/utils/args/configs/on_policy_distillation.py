from miles.utils.args.schema import A, Arg, BaseConfig


class OnPolicyDistillationConfig(BaseConfig):
    """Add on-policy distillation (OPD) related arguments.

    OPD is orthogonal to advantage estimators and can be applied on top of
    any estimator (GRPO, PPO, etc.) by adding a KL penalty to advantages.
    """

    use_opd: A[
        bool,
        Arg(help="Enable on-policy distillation (OPD). Must specify --opd-type when enabled."),
    ] = False
    opd_type: A[
        str | None,
        Arg(
            choices=["sglang", "megatron"],
            help=(
                "Type of on-policy distillation. "
                "'sglang': Teacher log-probs are obtained from external SGLang server during rollout. "
                "'megatron': Teacher model is loaded via --opd-teacher-load and forwarded during training."
            ),
        ),
    ] = None
    opd_kl_coef: A[
        float,
        Arg(help="On-policy distillation KL penalty coefficient. Default is 1.0."),
    ] = 1.0
    opd_log_prob_top_k: A[
        int,
        Arg(
            help=(
                "Number of top-k tokens to use for the re-think OPD token-level reward. "
                "Set to 0 to use sampled-token OPD."
            )
        ),
    ] = 0
    opd_top_k_strategy: A[
        str,
        Arg(
            choices=["only-student", "only-teacher", "intersection", "union", "xor"],
            help="Token set strategy for top-k OPD.",
        ),
    ] = "only-student"
    opd_reward_weight_mode: A[
        str,
        Arg(
            choices=["student_p", "teacher_p", "none"],
            help="Weighting scheme for top-k OPD token rewards.",
        ),
    ] = "student_p"
    opd_topk_per_position: A[
        bool,
        Arg(
            help=(
                "Send per-position token ids to the teacher/student scoring server "
                "(token_ids_logprob_positions) instead of the global top-k union, so the "
                "response is O(response_len * k) instead of O(response_len * |union|). "
                "Requires a patched sglang server that supports token_ids_logprob_positions; "
                "leave off for an unpatched server."
            )
        ),
    ] = False
    opd_teacher_urls: A[
        list[str] | None,
        Arg(
            type_parser=str,
            nargs="+",
            metavar="NAME=URL",
            help=(
                "Multi-teacher routing map for --opd-type=sglang, e.g. "
                "--opd-teacher-urls math=http://h1:30001/generate code=http://h2:30002/generate. "
                "Each sample is routed to the teacher named by "
                "sample.metadata[--opd-teacher-key]; the reserved name 'default' is the "
                "fallback for samples with a missing or unknown name. When unset, all "
                "samples are scored by the single teacher at --rm-url (original behavior)."
            ),
        ),
    ] = None
    opd_teacher_key: A[
        str,
        Arg(
            help=(
                "Sample metadata key holding the teacher name used for --opd-teacher-urls "
                "routing. Populated from the dataset's metadata column (see --metadata-key)."
            )
        ),
    ] = "opd_teacher"
    opd_teacher_load: A[
        str | None,
        Arg(
            help=(
                "The checkpoint for OPD teacher model. Required when --opd-type=megatron. "
                "The teacher model should have the same architecture as policy/ref model."
            )
        ),
    ] = None
    opd_teacher_ckpt_step: A[int | None, Arg(help="The checkpoint step for OPD teacher model.")] = None
