def compute_rollout_concurrency(args) -> int:
    if getattr(args, "rollout_endpoint_url", None) is not None:
        if args.async_max_concurrent_samples is not None:
            return args.async_max_concurrent_samples
        return args.rollout_batch_size * args.n_samples_per_prompt

    rollout_num_gpus = args.rollout_num_gpus or 0
    concurrency = args.sglang_server_concurrency * rollout_num_gpus // args.rollout_num_gpus_per_engine
    if args.eval_num_gpus > 0:
        concurrency += args.sglang_server_concurrency * args.eval_num_gpus // args.eval_num_gpus_per_engine
    if args.eval_uses_snapshots:
        concurrency = max(concurrency, args.sglang_server_concurrency)
    return concurrency


def get_rollout_url(args, endpoint: str) -> str:
    base_url = getattr(args, "rollout_endpoint_url", None)
    if base_url is None:
        base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
