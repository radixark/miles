def compute_rollout_concurrency(args) -> int:
    if args.rollout_endpoint_url is not None:
        if args.async_max_concurrent_samples is not None:
            return args.async_max_concurrent_samples
        return args.rollout_batch_size * args.n_samples_per_prompt

    concurrency = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
    if args.eval_num_gpus > 0:
        concurrency += args.sglang_server_concurrency * args.eval_num_gpus // args.eval_num_gpus_per_engine
    return concurrency


def get_rollout_url(args, endpoint: str) -> str:
    base_url = args.rollout_endpoint_url
    if base_url is None:
        base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
