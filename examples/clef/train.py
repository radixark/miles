"""Direct calibration training with a fresh Clef head and full Qwen weights.

Run under torchrun on a shared filesystem. This is a supervised FSDP2 training
entrypoint, not the token-generation RL actor: it consumes schema distributions
directly and never starts an inference router or generates reasoning.

Example (one node, externally prepared model/data):
    torchrun --standalone --nproc-per-node=8 -m examples.clef.train \
        --model-dir /models/Qwen3.8-27B --data-dir /data/calibration \
        --output-dir /scratch/clef/run --run-name 261003-clef-pilot
"""

import json
import os
import random
import time
import traceback
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from tap import Tap
from transformers import AutoProcessor

from examples.clef.checkpoint import load_checkpoint, save_checkpoint
from examples.clef.data import DecisionExample, LabeledRecord, augment_example, encode_example, file_sha256, read_examples
from examples.clef.joint_schema_model import collate_records
from examples.clef.model import TrainableClefModel, build_model, read_head_config, shard_model
from examples.clef.objective import decision_loss, prediction_rows, summarize
from examples.clef.resume import validate_resume_config
from examples.clef.telemetry import Telemetry


class Args(Tap):
    model_dir: str
    data_dir: str
    output_dir: str
    run_name: str
    checkpoint_dir: str = ""
    checkpoint_object_store: bool = False
    head_config: str = str(Path(__file__).with_name("joint_head_config.json"))
    global_batch_size: int = 64
    micro_batch_size: int = 1
    backbone_lr: float = 3e-7
    head_lr: float = 1e-5
    head_warmup_steps: int = 128
    epochs: int = 2
    max_steps: int = 0
    max_length: int = 65536
    multi_field_fraction: float = 0.25
    save_interval: int = 128
    eval_interval: int = 32
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    seed: int = 261003
    resume: str = ""
    wandb_project: str = ""
    wandb_entity: str = ""
    prometheus_port: int = 9090
    distributed_timeout_seconds: int = 3600
    forecastbench_path: str = ""
    allow_forecastbench_change: bool = False

    def process_args(self) -> None:
        if not 0 <= self.multi_field_fraction <= 1:
            raise ValueError("multi_field_fraction must be in [0,1]")
        if min(self.global_batch_size, self.micro_batch_size, self.epochs, self.max_length, self.save_interval, self.eval_interval) <= 0:
            raise ValueError("batch, length, epoch, and interval values must be positive")
        if self.head_warmup_steps < 0 or min(self.backbone_lr, self.head_lr, self.max_grad_norm) <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer or warmup configuration")
        if self.max_steps < 0:
            raise ValueError("max_steps must be nonnegative")
        if self.distributed_timeout_seconds <= 0:
            raise ValueError("distributed timeout must be positive")


def _gather_rows(rows: list[dict]) -> list[dict]:
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, rows)
    return [row for part in parts for row in part]


def _metrics(rows: list[dict], prefix: str) -> dict[str, float]:
    metrics = {f"{prefix}/{key}": value for key, value in summarize(rows).items()}
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    for source, group in groups.items():
        metrics.update({f"{prefix}/{source}/{key}": value for key, value in summarize(group).items()})
    return metrics


def _batch(labels: list[LabeledRecord], pad_token_id: int, device: torch.device) -> dict[str, Any]:
    return collate_records([label.encoded for label in labels], pad_token_id=pad_token_id, device=device)


def _clip_gradients(model: TrainableClefModel, max_norm: float, device: torch.device) -> float:
    local = [p.grad.to_local() for p in model.parameters() if p.grad is not None]
    squared = torch.zeros((), dtype=torch.float32, device=device)
    for gradient in local:
        squared += gradient.float().square().sum()
    dist.all_reduce(squared)
    norm = squared.sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("non-finite global gradient norm")
    coefficient = (max_norm / (norm + 1e-6)).clamp(max=1)
    for gradient in local:
        gradient.mul_(coefficient)
    return norm.item()


def _build_optimizer(model: TrainableClefModel, args: Args) -> torch.optim.Optimizer:
    backbone = [p for name, p in model.named_parameters() if not name.startswith("head.") and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr}, {"params": list(model.head.parameters()), "lr": args.head_lr},
    ], weight_decay=args.weight_decay, foreach=False)
    # Explicit zero-step states avoid checkpoint helpers lazily executing a dummy
    # Adam step for backbone parameters that have no gradients during head warmup.
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            optimizer.state[parameter] = {"step": torch.zeros(()), "exp_avg": torch.zeros_like(parameter), "exp_avg_sq": torch.zeros_like(parameter)}
    return optimizer


@torch.no_grad()
def _evaluate(
    model: TrainableClefModel, labels: list[LabeledRecord], pad_token_id: int, device: torch.device, output_dir: Path, step: int,
    prefix: str = "validation",
) -> dict[str, float]:
    model.eval()
    local_rows = []
    rank, world = dist.get_rank(), dist.get_world_size()
    # All ranks execute the same number of FSDP forwards, including the tail.
    for start in range(0, len(labels), world):
        index = start + rank
        label = labels[index if index < len(labels) else 0]
        logits = model(_batch([label], pad_token_id, device))
        if index < len(labels):
            local_rows.extend(prediction_rows(logits, [label]))
    rows = _gather_rows(local_rows)
    expected = {(label.encoded.record_id, q.question_id) for label in labels for q in label.encoded.questions}
    if len(rows) != len(expected) or {(row["id"], row["field_id"]) for row in rows} != expected:
        raise ValueError("validation coverage mismatch")
    if rank == 0:
        destination = output_dir / prefix / f"step_{step:07d}.jsonl"
        destination.parent.mkdir(exist_ok=True)
        destination.write_text("".join(json.dumps(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))
    model.train()
    metrics = _metrics(rows, prefix)
    if prefix == "forecastbench":
        # ForecastBench scores only the probability of Yes. Two-option summed
        # Brier is exactly twice this binary convention.
        if any(len(row["probabilities"]) != 2 or not row["hard_target"] for row in rows):
            raise ValueError("ForecastBench requires binary resolved targets")
        metrics[f"{prefix}/binary_brier"] = metrics[f"{prefix}/brier"] / 2
    return metrics


def _step_order(examples: list[DecisionExample], step: int, args: Args) -> list[int]:
    steps_per_epoch = len(examples) // args.global_batch_size
    in_warmup = step < args.head_warmup_steps
    phase_step = step if in_warmup else step - args.head_warmup_steps
    epoch, batch = divmod(phase_step, steps_per_epoch)
    order = list(range(len(examples)))
    random.Random(args.seed + epoch + (1_000_000 if in_warmup else 0)).shuffle(order)
    return order[batch * args.global_batch_size : (batch + 1) * args.global_batch_size]


def _train_step(
    model: TrainableClefModel, optimizer: torch.optim.Optimizer, examples: list[DecisionExample], processor: Any,
    step: int, args: Args, device: torch.device, trace: Any,
) -> tuple[list[dict], float, float, float]:
    start = time.monotonic()
    model.train_backbone = step >= args.head_warmup_steps
    optimizer.param_groups[0]["lr"] = args.backbone_lr if model.train_backbone else 0.0
    optimizer.zero_grad(set_to_none=True)
    indices = _step_order(examples, step, args)[dist.get_rank() :: dist.get_world_size()]
    accumulation = len(indices) // args.micro_batch_size
    rows, losses = [], []
    for micro_step in range(accumulation):
        selected = indices[micro_step * args.micro_batch_size : (micro_step + 1) * args.micro_batch_size]
        labels = [encode_example(
            processor.tokenizer,
            augment_example(examples[index], random.Random(args.seed + step * len(examples) + index), args.multi_field_fraction),
            args.max_length,
        ) for index in selected]
        # Reduce each microbatch into sharded gradients instead of retaining
        # full-model FP32 gradients during accumulation.
        model.set_requires_gradient_sync(True)
        logits = model(_batch(labels, processor.tokenizer.pad_token_id, device))
        loss = decision_loss(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite decision loss")
        (loss / accumulation).backward()
        losses.append(loss.detach())
        predictions = prediction_rows(logits, labels)
        rows.extend(predictions)
        for label, fields in zip(labels, logits, strict=True):
            prediction = [row for row in predictions if row["id"] == label.encoded.record_id]
            trace.write(json.dumps({"step": step + 1, "phase": "joint" if model.train_backbone else "head_warmup",
                                    "input_ids": label.encoded.input_ids, "prediction": prediction,
                                    "fields": [{"question_id": q.question_id, "option_ids": q.option_ids,
                                                "probabilities": scores.detach().float().softmax(-1).cpu().tolist(), "target": target}
                                               for q, scores, target in zip(label.encoded.questions, fields, label.targets, strict=True)]}) + "\n")
        del logits, loss
    norm = _clip_gradients(model, args.max_grad_norm, device)
    optimizer.step()
    trace.flush()
    loss_sum = torch.stack(losses).mean()
    dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
    torch.cuda.synchronize()
    return _gather_rows(rows), loss_sum.item(), norm, time.monotonic() - start


def _prepare(args: Args) -> tuple[list[DecisionExample], list[DecisionExample], dict[str, Any]]:
    root = Path(args.data_dir)
    train, validation = read_examples(root / "train.jsonl"), read_examples(root / "validation.jsonl")
    if not train or not validation or {e.record["id"] for e in train} & {e.record["id"] for e in validation}:
        raise ValueError("empty data or train/validation leakage")
    world = dist.get_world_size()
    if args.global_batch_size % (world * args.micro_batch_size) or len(train) % args.global_batch_size:
        raise ValueError("global batch must divide data and be divisible by world size * micro batch")
    config = args.as_dict()
    if args.forecastbench_path:
        forecast = read_examples(Path(args.forecastbench_path))
        if not forecast or {e.record["id"] for e in forecast} & {e.record["id"] for e in train + validation}:
            raise ValueError("empty ForecastBench or overlapping record IDs")
        config.update({"forecastbench_questions": len(forecast), "forecastbench_sha256": file_sha256(Path(args.forecastbench_path))})
    config.update({"world_size": world, "train_questions": len(train), "validation_questions": len(validation),
                   "train_sha256": file_sha256(root / "train.jsonl"), "validation_sha256": file_sha256(root / "validation.jsonl"),
                   "total_steps": args.max_steps or (args.head_warmup_steps + args.epochs * len(train) // args.global_batch_size),
                   "loss": "direct_brier", "reasoning": False, "lora": False, "precision": "fp32_master_bf16_compute"})
    output = Path(args.output_dir)
    if int(os.environ["LOCAL_RANK"]) == 0:
        output.mkdir(parents=True, exist_ok=bool(args.resume))
        (output / "config.json").write_text(json.dumps(config, indent=2))
        (output / "traces").mkdir(exist_ok=True)
    dist.barrier()
    return train, validation, config


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.distributed_timeout_seconds))
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed)
    train, validation, config = _prepare(args)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    head_config = read_head_config(Path(args.head_config))
    model = shard_model(build_model(args.model_dir, head_config, device), dist.get_world_size())
    optimizer = _build_optimizer(model, args)
    start = 0
    if args.resume:
        saved = load_checkpoint(model, optimizer, args.resume)
        validate_resume_config(saved["config"], config, allow_forecastbench_change=args.allow_forecastbench_change)
        start = saved["step"]
    labels = [encode_example(processor.tokenizer, example, args.max_length) for example in validation]
    forecast_labels = [encode_example(processor.tokenizer, example, args.max_length)
                       for example in read_examples(Path(args.forecastbench_path))] if args.forecastbench_path else []
    telemetry = Telemetry(Path(args.output_dir), args.run_name, config, args.wandb_project, args.wandb_entity, args.prometheus_port) if dist.get_rank() == 0 else None
    try:
        _run(model, optimizer, train, labels, processor, head_config, config, start, args, device, telemetry, forecast_labels)
    except BaseException:
        # Collective shutdown can deadlock after one rank fails. Print the
        # original error and exit so torchrun can terminate the other ranks.
        traceback.print_exc()
        os._exit(1)
    else:
        if telemetry is not None:
            telemetry.close()
        dist.destroy_process_group()


def _run(
    model: TrainableClefModel, optimizer: torch.optim.Optimizer, train: list[DecisionExample], labels: list[LabeledRecord],
    processor: Any, head_config: dict[str, int], config: dict[str, Any], start: int, args: Args, device: torch.device, telemetry: Telemetry | None,
    forecast_labels: list[LabeledRecord] | None = None,
) -> None:
    output = Path(args.output_dir)
    initial = _evaluate(model, labels, processor.tokenizer.pad_token_id, device, output, start)
    if forecast_labels:
        initial.update(_evaluate(model, forecast_labels, processor.tokenizer.pad_token_id, device, output, start, "forecastbench"))
    if telemetry is not None:
        telemetry.log(initial, start)
    trace_path = output / "traces" / f"rank_{dist.get_rank():04d}.jsonl"
    with trace_path.open("a") as trace:
        for step in range(start, config["total_steps"]):
            rows, loss, norm, seconds = _train_step(model, optimizer, train, processor, step, args, device, trace)
            metrics = {**_metrics(rows, "train"), "train/loss": loss, "train/grad_norm": norm,
                       "train/backbone_lr": optimizer.param_groups[0]["lr"], "train/head_lr": args.head_lr,
                       "perf/step_seconds": seconds, "perf/questions_per_second": args.global_batch_size / seconds}
            completed = step + 1
            if completed % args.eval_interval == 0 or completed == config["total_steps"]:
                metrics.update(_evaluate(model, labels, processor.tokenizer.pad_token_id, device, output, completed))
                if forecast_labels:
                    metrics.update(_evaluate(model, forecast_labels, processor.tokenizer.pad_token_id, device, output, completed, "forecastbench"))
            if telemetry is not None:
                telemetry.log(metrics, completed)
            if completed % args.save_interval == 0 or completed == config["total_steps"]:
                save_checkpoint(model, optimizer, output, completed, {"config": config}, processor, head_config,
                                checkpoint_root=args.checkpoint_dir or None, object_store=args.checkpoint_object_store)


if __name__ == "__main__":
    main()
