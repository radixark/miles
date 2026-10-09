"""Synchronous categorical GRPO on a trained Clef head, without token rollouts."""

import json
import os
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoProcessor

from examples.clef.checkpoint import load_checkpoint, save_checkpoint
from examples.clef.data import encode_example, file_sha256, read_examples
from examples.clef.model import build_model, read_head_config, shard_model
from examples.clef.objective import prediction_rows
from examples.clef.resume import validate_resume_config
from examples.clef.rl_objective import group_loss, sample_group, validate_targets
from examples.clef.telemetry import Telemetry
from examples.clef.train import Args as SupervisedArgs
from examples.clef.train import _batch, _build_optimizer, _clip_gradients, _evaluate, _gather_rows, _metrics, _prepare, _step_order


class Args(SupervisedArgs):
    head_warmup_steps: int = 0
    multi_field_fraction: float = 0.0
    backbone_lr: float = 1e-7
    head_lr: float = 1e-6
    group_size: int = 32
    record_reward_weight: float = 0.5
    clip_epsilon: float = 0.2
    brier_weight: float = 1.0
    kl_weight: float = 0.1
    freeze_backbone: bool = False

    def process_args(self) -> None:
        super().process_args()
        if self.head_warmup_steps or self.multi_field_fraction or self.micro_batch_size != 1:
            raise ValueError("RL uses fixed schemas, no warmup, and micro_batch_size=1")
        if self.group_size < 2 or not 0 <= self.record_reward_weight <= 1:
            raise ValueError("invalid group or reward weight")
        if not 0 < self.clip_epsilon < 1 or min(self.brier_weight, self.kl_weight) < 0:
            raise ValueError("invalid loss coefficients")


def disable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.0
        if hasattr(module, "attention_dropout") and isinstance(module.attention_dropout, (float, int)):
            module.attention_dropout = 0.0


def initial_model_identity(root: Path) -> dict[str, str]:
    """Content identity, not a path or mutable model name, protects resumes."""
    identity = {}
    if dist.get_rank() == 0:
        files = sorted(root.glob("*.safetensors"))
        if not files or not (root / "joint_head.safetensors").is_file():
            raise ValueError("a complete trained backbone/head export is required")
        identity = {path.name: file_sha256(path) for path in files}
        for path in sorted(root.iterdir()):
            if path.is_file() and path.name != "STAGED.json" and path.suffix in {".json", ".jinja", ".model"}:
                identity[path.name] = file_sha256(path)
    values = [identity]
    dist.broadcast_object_list(values, src=0)
    return values[0]


@torch.no_grad()
def reference_cache(model: Any, labels: list, pad: int, device: torch.device, output: Path, resume: bool) -> tuple[dict, str]:
    """Cache the initial policy once. Resume must reuse it, never recalculate it."""
    path = output / "reference.json"
    if resume:
        cache = json.loads(path.read_text())
    else:
        model.eval()
        rows = []
        rank, world = dist.get_rank(), dist.get_world_size()
        for start in range(0, len(labels), world):
            index = start + rank
            label = labels[index if index < len(labels) else 0]
            fields = model(_batch([label], pad, device))[0]
            validate_targets(label.targets, fields)
            if index < len(labels):
                rows.append({"id": label.encoded.record_id, "probabilities": [x.float().softmax(-1).cpu().tolist() for x in fields]})
        rows = _gather_rows(rows)
        cache = {row["id"]: row["probabilities"] for row in rows}
        # /scratch is node-local. Persist identical gathered probabilities on
        # every node, not only on global rank zero. Atomic replacement also
        # works when nodes happen to share the output filesystem.
        if int(os.environ["LOCAL_RANK"]) == 0:
            temporary = path.with_name(f".reference-rank{dist.get_rank()}.tmp")
            temporary.write_text(json.dumps(cache, sort_keys=True))
            temporary.replace(path)
        dist.barrier()
    if set(cache) != {label.encoded.record_id for label in labels}:
        raise ValueError("reference cache record coverage mismatch")
    model.train()
    digest = file_sha256(path)
    digests = [None] * dist.get_world_size()
    dist.all_gather_object(digests, digest)
    if len(set(digests)) != 1:
        raise ValueError("reference cache differs across nodes")
    return cache, digest


def train_step(model: Any, optimizer: Any, train: list, labels: list, reference: dict, processor: Any, step: int, args: Args, device: torch.device, trace: Any) -> tuple[list, dict]:
    started = time.monotonic()
    model.train_backbone = not args.freeze_backbone
    optimizer.zero_grad(set_to_none=True)
    indices = _step_order(train, step, args)[dist.get_rank() :: dist.get_world_size()]
    rows, observations = [], []
    for index in indices:
        label = labels[index]
        batch = _batch([label], processor.tokenizer.pad_token_id, device)
        with torch.no_grad():
            old = model(batch)[0]
            generator = torch.Generator(device=device).manual_seed(args.seed + step * len(train) + index)
            group = sample_group(old, label.targets, args.group_size, generator, args.record_reward_weight)
        model.set_requires_gradient_sync(True)
        fields = model(batch)[0]
        loss, metrics = group_loss(fields, label.targets, group, reference[label.encoded.record_id], clip_epsilon=args.clip_epsilon, brier_weight=args.brier_weight, kl_weight=args.kl_weight)
        (loss / len(indices)).backward()
        metrics["loss"] = loss.detach().item()
        observations.append(metrics)
        rows.extend(prediction_rows([fields], [label]))
        trace.write(
            json.dumps(
                {
                    "step": step + 1,
                    "id": label.encoded.record_id,
                    "input_ids": label.encoded.input_ids,
                    "option_ids": [q.option_ids for q in label.encoded.questions],
                    "probabilities": [x.detach().float().softmax(-1).cpu().tolist() for x in fields],
                    "actions": [x.cpu().tolist() for x in group.actions],
                    "rewards": group.rewards.cpu().tolist(),
                    "advantages": group.advantages.cpu().tolist(),
                    "old_log_probs": group.old_log_probs.cpu().tolist(),
                    "metrics": metrics,
                }
            )
            + "\n"
        )
    norm = _clip_gradients(model, args.max_grad_norm, device)
    optimizer.step()
    trace.flush()
    observations = _gather_rows(observations)
    metrics = {f"rl/{key}": sum(row[key] for row in observations) / len(observations) for key in observations[0]}
    torch.cuda.synchronize()
    metrics.update({"train/grad_norm": norm, "perf/step_seconds": time.monotonic() - started})
    return _gather_rows(rows), metrics


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.distributed_timeout_seconds))
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed)
    telemetry = None
    try:
        train, validation, config = _prepare(args)
        output = Path(args.output_dir)
        processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
        head_config = read_head_config(Path(args.model_dir) / "joint_head_config.json")
        config.update({"loss": "categorical_grpo_brier_kl", "temperature": 1.0, "head_config_values": head_config, "initial_head_sha256": file_sha256(Path(args.model_dir) / "joint_head.safetensors")})
        config["initial_model_identity"] = initial_model_identity(Path(args.model_dir))
        model = build_model(args.model_dir, head_config, device, pretrained_head=True)
        disable_dropout(model)
        if args.freeze_backbone:
            model.language_model.requires_grad_(False)
        model = shard_model(model, dist.get_world_size())
        optimizer = _build_optimizer(model, args)
        labels = [encode_example(processor.tokenizer, e, args.max_length) for e in train]
        val = [encode_example(processor.tokenizer, e, args.max_length) for e in validation]
        forecast = [encode_example(processor.tokenizer, e, args.max_length) for e in read_examples(Path(args.forecastbench_path))] if args.forecastbench_path else []
        reference, digest = reference_cache(model, labels, processor.tokenizer.pad_token_id, device, output, bool(args.resume))
        config["reference_sha256"] = digest
        start = 0
        if args.resume:
            saved = load_checkpoint(model, optimizer, args.resume)
            validate_resume_config(saved["config"], config, allow_forecastbench_change=args.allow_forecastbench_change)
            start = saved["step"]
        if int(os.environ["LOCAL_RANK"]) == 0:
            (output / "config.json").write_text(json.dumps(config, indent=2))
        if dist.get_rank() == 0:
            telemetry = Telemetry(output, args.run_name, config, args.wandb_project, args.wandb_entity, args.prometheus_port)
        metrics = _evaluate(model, val, processor.tokenizer.pad_token_id, device, output, start)
        if forecast:
            metrics.update(_evaluate(model, forecast, processor.tokenizer.pad_token_id, device, output, start, "forecastbench"))
        if telemetry:
            telemetry.log(metrics, start)
        with (output / "traces" / f"rank_{dist.get_rank():04d}.jsonl").open("a") as trace:
            for step in range(start, config["total_steps"]):
                rows, metrics = train_step(model, optimizer, train, labels, reference, processor, step, args, device, trace)
                metrics.update(_metrics(rows, "train"))
                metrics.update({"train/backbone_lr": 0.0 if args.freeze_backbone else args.backbone_lr, "train/head_lr": args.head_lr})
                completed = step + 1
                if completed % args.eval_interval == 0 or completed == config["total_steps"]:
                    metrics.update(_evaluate(model, val, processor.tokenizer.pad_token_id, device, output, completed))
                    if forecast:
                        metrics.update(_evaluate(model, forecast, processor.tokenizer.pad_token_id, device, output, completed, "forecastbench"))
                if telemetry:
                    telemetry.log(metrics, completed)
                if completed % args.save_interval == 0 or completed == config["total_steps"]:
                    save_checkpoint(model, optimizer, output, completed, {"config": config}, processor, head_config, checkpoint_root=args.checkpoint_dir or None, object_store=args.checkpoint_object_store)
    except BaseException:
        traceback.print_exc()
        os._exit(1)
    if telemetry:
        telemetry.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
