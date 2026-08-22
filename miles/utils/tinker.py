def uses_explicit_training_operations(args) -> bool:
    """Whether the Tinker protocol drives explicit training operations."""
    return bool(getattr(args, "tinker_backend", False))


def is_tinker_enabled(args) -> bool:
    """Whether the current Tinker-to-Multi-LoRA composition is enabled."""
    from miles.utils.multi_lora import uses_multi_lora_operation_executor

    return uses_multi_lora_operation_executor(args)


def validate_tinker_args(args) -> None:
    """Validate the Tinker adapter and select its queue-backed rollout path."""
    if not getattr(args, "tinker_backend", False):
        assert not getattr(args, "tinker_frontend", False), "--tinker-frontend requires --tinker-backend"
        assert not getattr(args, "tinker_api_key", None), "--tinker-api-key requires --tinker-frontend"
        return

    assert not (
        getattr(args, "tinker_api_key", None) and not getattr(args, "tinker_frontend", False)
    ), "--tinker-api-key requires --tinker-frontend (only the SDK frontend authenticates requests)"

    from miles.utils.environ import use_legacy_rollout_v1

    assert getattr(args, "multi_lora_n_adapters", 0) > 0, "--tinker-backend requires --multi-lora-n-adapters > 0"
    assert (
        not use_legacy_rollout_v1()
    ), "--tinker-backend needs the class-based rollout API (the default); unset MILES_USE_LEGACY_ROLLOUT_V1"
    if getattr(args, "tinker_frontend", False) and not getattr(args, "multi_lora_http_server_path", None):
        args.multi_lora_http_server_path = "miles.ray.tinker_frontend.http_server.TinkerFrontendHTTPServer"
    if args.rollout_function_path is None:
        args.rollout_function_path = "miles.rollout.multi_lora.rollout_fn.MultiLoraOperationBatchFn"
    if args.data_source_path == "miles.rollout.data_source.RolloutDataSourceWithBuffer":
        args.data_source_path = "miles.rollout.multi_lora.rollout_fn.TinkerNullDataSource"
    args.use_dynamic_global_batch_size = True
