from tests.fast.argument_snapshot.parse_outputs.results import ResultScenario


def result_scenarios() -> dict[str, ResultScenario]:
    scenarios = {backend: ResultScenario(backend=backend) for backend in ("megatron", "fsdp")}
    for backend in ("megatron", "fsdp"):
        for name, arguments in _shared_variants().items():
            scenarios[f"{backend}_{name}"] = ResultScenario(backend=backend, arguments=arguments)
        scenarios[f"{backend}_legacy"] = ResultScenario(backend=backend, legacy=True)
        scenarios[f"{backend}_custom_arguments"] = ResultScenario(
            backend=backend, custom=True, arguments=("--snapshot-custom", "29")
        )
        for name, (arguments, error, message) in _rejected_variants().items():
            scenarios[f"{backend}_reject_{name}"] = ResultScenario(
                backend=backend, arguments=arguments, error=error, message=message
            )

    for name, arguments in _megatron_variants().items():
        scenarios[f"megatron_{name}"] = ResultScenario(backend="megatron", arguments=arguments)
    scenarios["fsdp_hybrid_shard"] = ResultScenario(
        backend="fsdp", arguments=("--actor-num-gpus-per-node", "4", "--dp-replicate-size", "2")
    )
    scenarios["fsdp_yaml_cli_override"] = ResultScenario(
        backend="fsdp", arguments=("--config", "$FIXTURES/fsdp.yaml", "--lr", "0.000007")
    )
    scenarios["fsdp_yaml_unknown"] = ResultScenario(
        backend="fsdp", arguments=("--config", "$FIXTURES/invalid-fsdp.yaml"),
        error=ValueError, message="unknown key(s) in the YAML config",
    )
    scenarios["fsdp_reject_context_parallel"] = ResultScenario(
        backend="fsdp", arguments=("--context-parallel-size", "2"),
        error=AssertionError, message="Context parallelism is not supported",
    )
    scenarios["fsdp_reject_replica_divisibility"] = ResultScenario(
        backend="fsdp", arguments=("--actor-num-gpus-per-node", "3", "--dp-replicate-size", "2"),
        error=ValueError, message="must be divisible",
    )
    scenarios["megatron_reject_true_on_policy"] = ResultScenario(
        backend="megatron", arguments=("--true-on-policy-mode",),
        error=NotImplementedError, message="not supported on the megatron backend",
    )
    scenarios["fsdp_true_on_policy"] = ResultScenario(backend="fsdp", arguments=("--true-on-policy-mode",))
    scenarios["fsdp_prefill_logprobs"] = ResultScenario(
        backend="fsdp", arguments=("--true-on-policy-mode", "--recompute-logprobs-via-prefill")
    )
    scenarios["legacy_reject_fully_async"] = ResultScenario(
        backend="megatron", legacy=True, arguments=("--fully-async",),
        error=AssertionError, message="--fully-async needs the class-based rollout API",
    )
    return scenarios


def _shared_variants() -> dict[str, tuple[str, ...]]:
    return {
        "learning_rate": ("--lr", "0.000003"),
        "dynamic_batch": ("--use-dynamic-batch-size", "--max-tokens-per-gpu", "4096"),
        "dynamic_logprobs_override": (
            "--use-dynamic-batch-size", "--max-tokens-per-gpu", "4096", "--log-probs-max-tokens-per-gpu", "2048",
        ),
        "dynamic_global_batch": (
            "--use-dynamic-batch-size", "--max-tokens-per-gpu", "4096", "--use-dynamic-global-batch-size",
        ),
        "colocate": ("--colocate",),
        "colocate_without_offload": ("--colocate", "--no-offload-train", "--no-offload-rollout"),
        "offload_alias": ("--offload",),
        "debug_train": ("--debug-train-only",),
        "debug_rollout": ("--debug-rollout-only", "--rollout-num-gpus", "1"),
        "debug_optimizer": ("--debug-disable-optimizer",),
        "load_rollout": ("--load-debug-rollout-data", "$FIXTURES/rollout.pt"),
        "save": ("--save", "$FIXTURES/save", "--save-interval", "2"),
        "dump_details": ("--dump-details", "$FIXTURES/dump"),
        "explicit_event_directory": ("--save", "$FIXTURES/save", "--save-debug-event-data", "$FIXTURES/events"),
        "load_fallback": ("--load", "$FIXTURES/missing", "--ref-load", "$FIXTURES/ref", "--ref-ckpt-step", "7"),
        "load_existing": ("--load", "$FIXTURES/checkpoint"),
        "custom_yaml": ("--custom-config-path", "$FIXTURES/custom.yaml", "--lr", "0.000003"),
        "eval_legacy_path": ("--eval-prompt-data", "$FIXTURES/data.jsonl", "--eval-interval", "2"),
        "eval_named_paths": ("--eval-prompt-data", "first", "$FIXTURES/data.jsonl", "second", "$FIXTURES/data2.jsonl"),
        "eval_yaml": ("--eval-config", "$FIXTURES/eval.yaml", "--eval-interval", "2"),
        "eval_fleet": (
            "--eval-num-gpus", "1", "--eval-num-gpus-per-engine", "1",
            "--eval-prompt-data", "$FIXTURES/data.jsonl", "--eval-interval", "2",
            "--eval-hf-dir", "$FIXTURES/eval-hf", "--eval-sglang-enable-metrics",
        ),
        "reward_and_clip_defaults": ("--reward-key", "score", "--eps-clip", "0.3"),
        "reward_and_clip_overrides": ("--reward-key", "score", "--eval-reward-key", "eval_score", "--eps-clip-high", "0.4"),
        "context_length": ("--rollout-max-context-len", "1024"),
        "context_length_overrides": ("--rollout-max-context-len", "1024", "--rollout-max-prompt-len", "512", "--eval-max-context-len", "2048"),
        "single_sample": ("--n-samples-per-prompt", "1"),
        "multiple_steps": ("--rollout-batch-size", "4", "--n-samples-per-prompt", "2", "--num-steps-per-rollout", "2"),
        "oversampling": ("--over-sampling-batch-size", "4"),
        "fully_async": ("--fully-async",),
        "fully_async_in_place": ("--fully-async", "--pause-generation-mode", "in_place"),
        "fully_async_without_cache_namespace": ("--fully-async", "--pause-generation-mode", "in_place", "--no-namespaced-radix-cache"),
        "session_v1": ("--use-session-server",),
        "session_v2": ("--use-session-server", "v2"),
        "session_router_override": ("--use-session-server", "--sglang-router-policy", "round_robin"),
        "router_ipv6": ("--sglang-router-ip", "::1"),
        "sglang_dp_attention": ("--sglang-dp-size", "2", "--sglang-enable-dp-attention", "--rollout-num-gpus-per-engine", "2"),
        "sglang_tp_derived": ("--sglang-tp-size", "8", "--rollout-num-gpus-per-engine", "2"),
        "rollout_ft": ("--use-fault-tolerance",),
        "ft_explicit_api": ("--use-fault-tolerance", "--api-server-port", "23456", "--no-mini-ft-controller-enable"),
        "opd_external": ("--use-opd", "--opd-type", "sglang"),
        "rollout_logprobs": ("--use-rollout-logprobs",),
        "ci": ("--ci-test", "--no-enable-sample-ownership-checker", "--save-debug-event-data", "$FIXTURES/events"),
        "kubernetes": ("--cluster-backend", "kubernetes", "--mooncake-store-init-kwargs", '{"master_server_address":"localhost:50051"}'),
        "ray_rpc": ("--worker-comm-backend", "rpc"),
        "external_rollout": ("--rollout-external-engine-addrs", "127.0.0.1:30000"),
        "lora_targets": ("--lora-rank", "8", "--target-modules", "q_proj,v_proj", "--exclude-modules", "v_proj"),
    }


def _megatron_variants() -> dict[str, tuple[str, ...]]:
    return {
        "train_ft": ("--use-fault-tolerance", "--ft-components", "train", "--use-dynamic-batch-size", "--max-tokens-per-gpu", "4096"),
        "indep_dp": ("--indep-dp", "--use-dynamic-batch-size", "--max-tokens-per-gpu", "4096", "--actor-num-gpus-per-node", "2"),
        "ppo": ("--advantage-estimator", "ppo"),
        "ppo_save": ("--advantage-estimator", "ppo", "--save", "$FIXTURES/save", "--critic-lr", "0.000001"),
        "offload_disk": ("--offload-train", "--offload-train-target", "disk", "--offload-train-disk-dir", "$FIXTURES/offload"),
        "bridge_load_fallback": ("--megatron-to-hf-mode", "bridge", "--load", "$FIXTURES/missing", "--ref-load", "$FIXTURES/ref"),
        "moe_dispatch": ("--moe-token-dispatcher-type", "allgather"),
        "fully_async_colocate": ("--fully-async", "--colocate", "--pause-generation-mode", "retract"),
        "reinforce": ("--advantage-estimator", "reinforce_plus_plus", "--normalize-advantages"),
    }


def _rejected_variants() -> dict[str, tuple[tuple[str, ...], type[Exception], str]]:
    return {
        "logprobs_tis": (("--use-rollout-logprobs", "--use-tis"), AssertionError, "cannot be set at the same time"),
        "dynamic_tokens": (("--use-dynamic-batch-size",), AssertionError, "max_tokens_per_gpu must be set"),
        "dynamic_global": (("--use-dynamic-global-batch-size",), AssertionError, "requires --use-dynamic-batch-size"),
        "eval_without_data": (("--eval-interval", "2"), AssertionError, "Evaluation datasets must be configured"),
        "eval_overrides_without_fleet": (("--eval-sglang-enable-metrics",), AssertionError, "needs --eval-num-gpus > 0"),
        "eval_odd_paths": (("--eval-prompt-data", "a", "b", "c"), ValueError, "name/path pairs"),
        "eval_empty_yaml": (("--eval-config", "$FIXTURES/empty-eval.yaml"), ValueError, "does not define any datasets"),
        "save_without_directory": (("--save-interval", "2"), AssertionError, "'--save' is required"),
        "sentinel_without_directory": (("--save-trigger-sentinel", "$FIXTURES/sentinel"), AssertionError, "'--save' is required"),
        "session_version": (("--use-session-server", "v3"), ValueError, "not a known session server version"),
        "session_partial": (("--use-session-server", "--partial-rollout"), AssertionError, "does not support --partial-rollout"),
        "session_v2_group_rm": (("--use-session-server", "v2", "--group-rm"), ValueError, "does not support --group-rm"),
        "prefill_without_on_policy": (("--recompute-logprobs-via-prefill",), AssertionError, "requires --true-on-policy-mode"),
        "opd_without_type": (("--use-opd",), ValueError, "--opd-type must be specified"),
        "opd_negative_topk": (("--use-opd", "--opd-type", "sglang", "--opd-log-prob-top-k", "-1"), ValueError, "must be non-negative"),
        "opd_teacher_without_opd": (("--opd-teacher-load", "$FIXTURES/ref"), ValueError, "--use-opd is not enabled"),
        "opd_megatron_without_checkpoint": (("--use-opd", "--opd-type", "megatron"), ValueError, "--opd-teacher-load is required"),
        "prompt_context_length": (("--rollout-max-context-len", "16", "--rollout-max-prompt-len", "16"), AssertionError, "must be smaller"),
        "oversampling": (("--over-sampling-batch-size", "1"), AssertionError, "should be greater than or equal"),
        "async_concurrency": (("--n-samples-per-prompt", "2", "--async-max-concurrent-samples", "1"), AssertionError, "must be at least"),
        "fully_async_partial": (("--fully-async", "--partial-rollout"), AssertionError, "does not support --partial-rollout"),
        "fully_async_abort": (("--fully-async", "--pause-generation-mode", "abort"), AssertionError, "cannot use --pause-generation-mode abort"),
        "p2p_colocate": (("--update-weight-transfer-mode", "p2p", "--colocate"), AssertionError, "not compatible with --colocate"),
        "disk_delta_colocate": (("--update-weight-transfer-mode", "disk-delta", "--colocate"), AssertionError, "not compatible with --colocate"),
        "lora_without_targets": (("--lora-rank", "8"), AssertionError, "'--target-modules' is required"),
        "mini_ft_without_api": (("--mini-ft-controller-enable", "--api-server-port", "0"), ValueError, "requires --api-server-port"),
        "external_pd_without_engines": (("--rollout-external-router-pd",), AssertionError, "only applies to external rollout engines"),
    }
