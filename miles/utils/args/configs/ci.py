from miles.utils.args.schema import A, Arg, BaseConfig


class CiConfig(BaseConfig):
    ci_disable_config_snapshot: A[bool, Arg()] = False
    config_snapshot_name: A[str | None, Arg()] = None
    ci_enable_metrics_capture: bool

    ci_inject_missing_prefetched_batch_bug: A[
        bool, Arg(help="Discard the restored prefetched batch to test sample ownership failure detection.")
    ] = False
    ci_test: A[bool, Arg()] = False
    ci_disable_kl_checker: A[bool, Arg()] = False
    ci_disable_logprobs_checker: A[bool, Arg()] = False
    ci_disable_weight_update_checker: A[bool, Arg()] = False
    ci_metric_checker_key: A[str | None, Arg()] = None
    ci_metric_checker_threshold: A[float | None, Arg()] = None
    ci_metric_checker_expect_num: A[
        int | None, Arg(help="Require exactly this many eval checks, all meeting the CI threshold.")
    ] = None
    ci_assert_prefill_lag_max: A[
        int | None,
        Arg(help="Require every rollout's prompt KV to lag its decode weight version by at most this much."),
    ] = None
    ci_save_grad_norm: A[str | None, Arg()] = None
    ci_load_grad_norm: A[str | None, Arg()] = None
    ci_save_model_hash: A[bool, Arg()] = False
    ci_check_model_hash: A[bool, Arg()] = False
