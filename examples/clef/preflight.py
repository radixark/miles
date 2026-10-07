"""Distributed numerical/gradient checks; never optimize the real backbone."""

import json
import os
from pathlib import Path
from typing import Any

import fsspec
import torch
import torch.distributed as dist
from tap import Tap
from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig, Qwen3_5VisionConfig

from examples.clef.checkpoint import load_checkpoint, save_checkpoint
from examples.clef.data import DecisionExample, LabeledRecord, encode_example, read_examples
from examples.clef.joint_schema_model import EncodedQuestion, EncodedRecord, JointSchemaHead, collate_records, load_release_model
from examples.clef.model import TrainableClefModel, build_model, read_head_config, shard_model
from examples.clef.objective import decision_loss
from examples.clef.train import Args as TrainArgs
from examples.clef.train import _build_optimizer


class Args(Tap):
    model_dir: str
    output_dir: str
    real_model: bool = False
    checkpoint_dir: str = ""
    checkpoint_object_store: bool = False
    data_path: str = ""
    allocate_optimizer: bool = False


def _tiny_model(device: torch.device) -> tuple[TrainableClefModel, dict[str, int], LabeledRecord]:
    text = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0, "mrope_section": [2, 3, 3]},
    )
    vision = Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=64)
    config = Qwen3_5Config(text_config=text.to_dict(), vision_config=vision.to_dict(), image_token_id=120, video_token_id=121, vision_start_token_id=122, vision_end_token_id=123)
    backbone = Qwen3_5ForConditionalGeneration(config).to(device)
    backbone.model.visual.requires_grad_(False)
    head_config = {"hidden_size": 64, "width": 32, "routing_layers": 1, "layers": 1, "heads": 4, "feedforward": 64}
    model = TrainableClefModel(backbone, JointSchemaHead(**head_config).to(device))
    fields = (
        EncodedQuestion("answer", 1, (0, 3), ((3, 6), (6, 9)), ("A", "B")),
        EncodedQuestion("candidate", 0, (9, 11), ((11, 13), (13, 15)), ("true", "false")),
    )
    label = LabeledRecord(EncodedRecord(tuple(range(20, 36)), fields, "tiny-probe"), ((0.7, 0.3), (0.7, 0.3)), "coin")
    return model, head_config, label


def _real_model(args: Args, processor: Any, device: torch.device) -> tuple[TrainableClefModel, dict[str, int], LabeledRecord]:
    head_config = read_head_config(Path(__file__).with_name("joint_head_config.json"))
    model = build_model(args.model_dir, head_config, device)
    example = DecisionExample(
        {"id": "real-backbone-probe", "state": "A coin lands heads with probability 0.7. Take a guess at the next flip.",
         "questions": {"answer": {"type": "choice", "instructions": "Choose the future outcome.", "criteria": {"A": "Heads", "B": "Tails"}}}},
        {"answer": {"A": 0.7, "B": 0.3}}, "coin",
    )
    if args.data_path:
        examples = read_examples(Path(args.data_path))
        example = max(examples, key=lambda row: len(json.dumps(row.record)))
    return model, head_config, encode_example(processor.tokenizer, example, max_length=65536)


def _check_gradients(model: TrainableClefModel, label: LabeledRecord, device: torch.device) -> dict[str, Any]:
    batch = collate_records([label.encoded], pad_token_id=0, device=device)
    counts = {}
    model.train()
    for enabled in (False, True):
        model.zero_grad(set_to_none=True)
        model.train_backbone = enabled
        loss = decision_loss(model(batch), [label])
        loss.backward()
        missing, bad, backbone_gradients, head_gradients = [], [], 0, 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            required = enabled or name.startswith("head.")
            if parameter.grad is None:
                if required:
                    missing.append(name)
                continue
            gradient = parameter.grad.to_local()
            if not torch.isfinite(gradient).all():
                bad.append(name)
            backbone_gradients += not name.startswith("head.")
            head_gradients += name.startswith("head.")
        if missing or bad or (not enabled and backbone_gradients):
            raise AssertionError({"phase": enabled, "missing": missing, "nonfinite": bad, "backbone_gradients": backbone_gradients})
        counts["joint" if enabled else "head_warmup"] = {"loss": loss.item(), "head_gradient_tensors": head_gradients, "backbone_gradient_tensors": backbone_gradients}
    model.zero_grad(set_to_none=True)
    return counts


def _checkpoint_probe(
    model: TrainableClefModel, head_config: dict[str, int], label: LabeledRecord, processor: Any, output: Path, device: torch.device, checkpoint_dir: str, object_store: bool,
) -> dict[str, float]:
    # A tiny randomly initialized model is a unit-test fixture, not a training run.
    optimizer = _build_optimizer(model, TrainArgs().from_dict({"model_dir": "fixture", "data_dir": "fixture", "output_dir": "fixture", "run_name": "fixture"}))
    model.eval()
    batch = collate_records([label.encoded], pad_token_id=0, device=device)
    with torch.no_grad():
        expected = model(batch)[0][0].float().softmax(-1)
    save_checkpoint(model, optimizer, output, 0, {"config": {}}, processor, head_config, checkpoint_root=checkpoint_dir or None, object_store=object_store)
    root = checkpoint_dir.rstrip("/") + "/step_0000000" if checkpoint_dir else output / "checkpoints" / "step_0000000"
    # Changing the head tests actual DCP restoration rather than loading untouched weights.
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.add_(0.1)
        for state in optimizer.state.values():
            state["exp_avg"].fill_(0.2)
            state["exp_avg_sq"].fill_(0.3)
    load_checkpoint(model, optimizer, root)
    with torch.no_grad():
        restored = model(batch)[0][0].float().softmax(-1)
    torch.testing.assert_close(expected, restored, atol=1e-6, rtol=1e-5)
    if any(torch.count_nonzero(state["exp_avg"].to_local()) or torch.count_nonzero(state["exp_avg_sq"].to_local()) for state in optimizer.state.values()):
        raise AssertionError("checkpoint did not restore optimizer moments")
    result = {"native_restore_max_error": (expected - restored).abs().max().item()}
    if dist.get_rank() == 0:
        export = output / "downloaded-hf" if checkpoint_dir.startswith("s3://") else Path(root) / "hf"
        if checkpoint_dir.startswith("s3://"):
            fs, key = fsspec.core.url_to_fs(str(root) + "/hf")
            fs.get(key, str(export), recursive=True)
        serving, _ = load_release_model(export, device=device)
        with torch.no_grad():
            exported = serving(batch)[0][0].float().softmax(-1)
        torch.testing.assert_close(expected, exported, atol=0.005, rtol=0.005)
        result["serving_probability_max_error"] = (expected - exported).abs().max().item()
    dist.barrier()
    return result


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.manual_seed(73)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model, head_config, label = _real_model(args, processor, device) if args.real_model else _tiny_model(device)
    model = shard_model(model, dist.get_world_size())
    optimizer = _build_optimizer(model, TrainArgs().from_dict({"model_dir": args.model_dir, "data_dir": "preflight", "output_dir": args.output_dir, "run_name": "preflight"})) if args.allocate_optimizer else None
    result = {"real_backbone": args.real_model, "world_size": dist.get_world_size(), "gradients": _check_gradients(model, label, device)}
    result.update({"record_id": label.encoded.record_id, "input_tokens": len(label.encoded.input_ids),
                   "fields": len(label.encoded.questions), "optimizer_allocated": optimizer is not None})
    output = Path(args.output_dir)
    if not args.real_model:
        result.update(_checkpoint_probe(model, head_config, label, processor, output, device, args.checkpoint_dir, args.checkpoint_object_store))
    result["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    if dist.get_rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "preflight.json").write_text(json.dumps(result, indent=2))
        print("PREFLIGHT_PASS", json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
