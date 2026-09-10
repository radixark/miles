"""Standalone #3170-style capacity probe; no Miles runtime files are modified.

The launcher selects this program for a one-slot trainer-only job. It warms up
forward/backward + Adam, measures a second maximum-token pass on every DP
replica, reports the worst-rank estimate, and exits. It never launches engines
or users. The serving job subsequently receives a plain integer slot count.

Example:
    python examples/multi_lora/run_pressure.py --mode probe --output-dir /shared/probe

Recalculate from the saved measurements without allocating any GPUs:
    python -m examples.multi_lora.slot_capacity --from-raw /shared/probe/slot-probe-raw.json \
        --output /shared/capacity.json
"""

import argparse
import asyncio
import json
import math
import sys
import uuid
from pathlib import Path

_ADAM = dict(learning_rate=0.0, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0)


def _write_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _report(raw, *, margin_bytes, host_budget_bytes, keep_k):
    if margin_bytes < 0 or host_budget_bytes < 0 or keep_k < 1:
        raise ValueError("budgets must be nonnegative and keep_k must be positive")
    ranks = []
    precision = raw["precision"]
    weight = 2 if precision["bf16"] else 4
    grad = 4 if precision["fp32_grad"] else weight
    # LayerWise distributes whole optimizer parameters over DP ranks. Model and
    # gradient copies are replicated over DP, but master/moments are not.
    for baseline, after in zip(raw["measure_before"], raw["after"], strict=True):
        measured = after["resident_slot_bytes"]
        predicted = after["adapter_local_params"] * (weight + grad)
        predicted += after["optimizer_local_params"] * ((4 if precision["bf16"] else 0) + 8)
        if min(measured, predicted) <= 0:
            raise ValueError("probe must measure positive slot storage and parameter counts on every rank")
        headroom = after["free"] + measured
        activation = max(0, after["max_allocated"] - baseline["allocated"])
        available = max(0, headroom - activation - margin_bytes)
        ranks.append(
            {
                "rank": after["rank"],
                "headroom_bytes": headroom,
                "activation_peak_bytes": activation,
                "measured_slot_bytes": measured,
                "predicted_slot_bytes": predicted,
                "n_measured": available // measured,
                "n_theoretical": available // predicted,
            }
        )
    if not ranks:
        raise ValueError("no ranks in probe report")
    full_counts = {rank["adapter_full_params"] for rank in raw["after"]}
    if len(full_counts) != 1 or min(full_counts) <= 0:
        raise ValueError("inconsistent unsharded adapter parameter counts")
    version_bytes = full_counts.pop() * weight
    n_host = host_budget_bytes // (keep_k * version_bytes) if host_budget_bytes else None
    n_gpu = min(rank["n_measured"] for rank in ranks)
    n_slots = min(n_gpu, n_host) if n_host is not None else n_gpu
    return {
        "schema_version": 1,
        "status": "estimated",
        "n_slots": n_slots,
        "n_theoretical_trainer": min(rank["n_theoretical"] for rank in ranks),
        "n_measured_trainer": n_gpu,
        "n_engine_host": n_host,
        "binding_constraint": "engine host RAM" if n_host is not None and n_host < n_gpu else "trainer GPU",
        "worst_rank": min(ranks, key=lambda rank: rank["n_measured"])["rank"],
        "keep_k": keep_k,
        "engine_adapter_version_bytes": version_bytes,
        "train_memory_margin_bytes": margin_bytes,
        "engine_host_lora_budget_bytes": host_budget_bytes,
        "engine_gpu_capacity_checked": False,
        "e2e_max_n": None,
        **raw["workload"],
        "ranks": ranks,
    }


async def _snapshots(trainer, phase):
    # Existing generic dispatch reaches only the probe subclass's read-only
    # measurement method. No method is added to TrainerController or MilesBackend.
    return await trainer._execute_slots("pressure_memory_snapshot", phase=phase)


async def _probe(args, trainer, backend):
    before = await _snapshots(trainer, "before")
    world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    if len(before) != world_size:
        raise ValueError(f"missing rank measurements: {len(before)} != {world_size}")
    dp_sizes = {snapshot["data_parallel_size"] for snapshot in before}
    if len(dp_sizes) != 1:
        raise ValueError(f"inconsistent DP sizes: {dp_sizes}")
    tokens = args.max_tokens_per_gpu
    # Every DP replica gets a full row, not an empty or padded shard.
    row = {"tokens": [1] * tokens, "target_len": tokens - 1, "weights": [1.0] * (tokens - 1)}
    datums = [(0, row) for _ in range(dp_sizes.pop())]
    raw = {
        "workload": {
            "hf_checkpoint": args.hf_checkpoint,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules,
            "max_tokens_per_gpu": tokens,
            "trainer_gpus": world_size,
            "training_tp": args.tensor_model_parallel_size,
            "training_ep": args.expert_model_parallel_size,
        },
        "precision": {"bf16": args.bf16, "fp32_grad": args.accumulate_allreduce_grads_in_fp32},
        "before": before,
    }
    raw_path = args.capacity_output.with_name("slot-probe-raw.json")
    _write_json(raw_path, raw)
    await backend.load_slot(0, args.lora_rank, float(args.lora_alpha or 2 * args.lora_rank))
    try:
        for batch_id, phase in [(-1, "warmup"), (-2, "after")]:
            if phase == "after":
                raw["measure_before"] = await _snapshots(trainer, "measure")
            await backend.forward_backward(batch_id, datums, "cross_entropy", {})
            outcomes = await backend.optim_step({0: _ADAM})
            norm = outcomes.get(0, {}).get("grad_norm")
            if norm is None or not math.isfinite(norm):
                raise RuntimeError(f"probe optimizer step failed: {outcomes}")
            raw[phase] = await _snapshots(trainer, phase)
            _write_json(raw_path, raw)
    finally:
        await backend.unload_slot(0)
    report = _report(
        raw,
        margin_bytes=args.capacity_margin_bytes,
        host_budget_bytes=args.engine_host_lora_budget_bytes,
        keep_k=args.engine_keep_k,
    )
    _write_json(args.capacity_output, report)
    print(json.dumps(report, indent=2), flush=True)
    if report["n_slots"] < 1:
        raise RuntimeError("No slot fits the selected headroom and budget")


async def _measure(args):
    # GPU/Ray dependencies stay out of the offline report calculator.
    import ray
    from ray.util.placement_group import remove_placement_group

    from miles.ray.placement_group import create_placement_groups
    from miles.ray.specs.train import specs_trainer
    from miles.ray.train.group import TrainerController
    from miles.tinker.runtime import MilesBackend
    from miles.utils import object_store
    from miles.utils.audit_utils.process_identity import MainProcessIdentity
    from miles.utils.http_utils import init_http_client
    from miles.utils.logging_utils import configure_logger
    from miles.utils.workers.ray_worker_manager import RayWorkerManager

    if not args.multi_lora or args.multi_lora_n_adapters != 1 or args.use_critic:
        raise ValueError("probe requires one multi-LoRA actor slot and no critic")
    if args.fp16 or getattr(args, "use_precision_aware_optimizer", False):
        raise ValueError("probe cross-check supports the standard BF16/FP32 LayerWise Adam configuration")
    if args.max_tokens_per_gpu < 2:
        raise ValueError("probe needs at least two tokens per GPU")
    args.capacity_output.parent.mkdir(parents=True, exist_ok=True)
    if args.capacity_output.exists():
        raise FileExistsError(args.capacity_output)
    configure_logger(args, source=MainProcessIdentity())
    init_http_client(args)
    ray.init(address="auto", namespace=f"pressure-probe-{uuid.uuid4().hex}")
    pgs, manager, trainer = {}, None, None
    try:
        # Reserve the intended placement, but create trainer workers only.
        pgs = create_placement_groups(args)
        specs = [
            spec.model_copy(update={"worker_class": "examples.multi_lora.slot_probe_actor.SlotProbeActor"})
            for spec in specs_trainer(args)
        ]
        manager = RayWorkerManager.launch(specs, pgs)
        object_store.init_instance(args, contribute_segment=False)
        trainer = TrainerController(
            args=args,
            role="actor",
            with_ref=False,
            with_opd_teacher=False,
            inference_controller=None,
            rollout_executor=None,
        )
        await trainer.init()
        dp_size = args.actor_num_nodes * args.actor_num_gpus_per_node // args.tensor_model_parallel_size
        await _probe(args, trainer, MilesBackend(trainer, "", dp_size=dp_size))
    finally:
        if trainer is not None:
            await trainer.dispose()
        if manager is not None:
            cells = await manager.get_cell_infos.remote(pool_ids=["trainer-actor"])
            await manager.stop_cells.remote(list(cells))
            ray.kill(manager)
        for pg in {entry.pg for entry in pgs.values()}:
            remove_placement_group(pg)
        ray.shutdown()


def _add_probe_args(parser):
    parser.add_argument("--capacity-output", type=Path, required=True)
    parser.add_argument("--capacity-margin-bytes", type=int, default=2 * 1024**3)
    parser.add_argument(
        "--engine-host-lora-budget-bytes",
        type=int,
        default=0,
        help="RAM budget per engine, not per node; 0 leaves this constraint unchecked",
    )
    parser.add_argument("--engine-keep-k", type=int, default=2)
    return parser


def _main():
    if any(arg.split("=", 1)[0] == "--from-raw" for arg in sys.argv[1:]) or sys.argv[1:] in ([], ["--help"]):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--from-raw", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--margin-bytes", type=int, default=2 * 1024**3)
        parser.add_argument("--host-budget-bytes", type=int, default=0)
        parser.add_argument("--keep-k", type=int, default=2)
        options = parser.parse_args()
        report = _report(
            json.loads(options.from_raw.read_text()),
            margin_bytes=options.margin_bytes,
            host_budget_bytes=options.host_budget_bytes,
            keep_k=options.keep_k,
        )
        options.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(options.output, report)
        print(json.dumps(report, indent=2))
        return
    from miles.utils.arguments import parse_args

    args = parse_args(add_custom_arguments=_add_probe_args, entry="serve")
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True
    asyncio.run(_measure(args))


if __name__ == "__main__":
    _main()
