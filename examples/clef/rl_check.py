"""Executable objective and real-head fixture checks; no production training."""

import io
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from tap import Tap

from examples.clef.data import LabeledRecord
from examples.clef.joint_schema_model import collate_records
from examples.clef.model import load_trained_head, shard_model
from examples.clef.preflight import _tiny_model
from examples.clef.rl_objective import DecisionGroup, group_loss, sample_group
from examples.clef.rl_train import Args as TrainArgs
from examples.clef.rl_train import reference_cache, train_step
from miles.backends.fsdp_utils.checkpoint import ModelState, OptimizerState


class Args(Tap):
    device: str = "cpu"
    distributed: bool = False


def main() -> None:
    args = Args().parse_args()
    device = torch.device(args.device)
    if args.distributed:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", torch.cuda.current_device())
        dist.init_process_group("nccl")
    generator = torch.Generator(device=device).manual_seed(42)
    scores = torch.zeros(3, device=device, requires_grad=True)
    target = [[0.0, 1.0, 0.0]]
    group = sample_group([scores], target, 4096, generator)
    loss, metrics = group_loss([scores], target, group, [[1 / 3] * 3], brier_weight=0, kl_weight=0)
    loss.backward()
    assert scores.grad[1] < 0 and scores.grad[[0, 2]].min() > 0
    assert abs(metrics["reference_kl"]) < 1e-6
    multi = sample_group([scores, torch.zeros(2, device=device)], [target[0], [1.0, 0.0]], 32, generator)
    correctness = torch.stack([multi.actions[0] == 1, multi.actions[1] == 0])
    expected = 0.5 * correctness.float().mean(0) + 0.5 * correctness.all(0).float()
    assert torch.equal(multi.rewards, expected)
    constant = DecisionGroup(group.actions, group.old_log_probs, torch.ones_like(group.rewards), torch.zeros_like(group.advantages))
    scores.grad = None
    loss, _ = group_loss([scores], target, constant, [[1 / 3] * 3], brier_weight=0, kl_weight=0)
    loss.backward()
    assert scores.grad.abs().max() == 0
    changed = torch.tensor([-2.0, 2.0, -2.0], device=device, requires_grad=True)
    _, clipped = group_loss([changed], target, group, [[1 / 3] * 3])
    assert clipped["clip_fraction"] > 0.9 and clipped["reference_kl"] > 0
    try:
        sample_group([scores], [[0.2, 0.6, 0.2]], 32, generator)
    except ValueError:
        pass
    else:
        raise AssertionError("soft labels accepted by RL")
    model, _, label = _tiny_model(device)
    targets = [[1.0, 0.0], [0.0, 1.0]]
    batch = collate_records([label.encoded], pad_token_id=0, device=device)
    with torch.no_grad():
        fields = model(batch)[0]
        reference = [x.float().softmax(-1).tolist() for x in fields]
        group = sample_group(fields, targets, 32, generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.head.prior_logit_scale.detach().clone()
    loss, _ = group_loss(model(batch)[0], targets, group, reference)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.language_model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.head.prior_logit_scale)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "head.safetensors"
        state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.head.state_dict().items()}
        for name in ("prior_logit_scale", "joint_logit_scale", "residual_gate"):
            state[name] = state[name].reshape(())
        save_file(state, path)
        restored, _, _ = _tiny_model(device)
        load_trained_head(restored.head, path)
        assert all(torch.equal(value, restored.head.state_dict()[name]) for name, value in model.head.state_dict().items())
    if args.distributed:
        model, _, label = _tiny_model(device)
        model = shard_model(model, dist.get_world_size())
        cache_root = Path("/scratch") / f"2dcf7753-rl-fixture-{os.environ['MASTER_PORT']}"
        if dist.get_rank() == 0:
            cache_root.mkdir(exist_ok=False)
        dist.barrier()
        hard_label = LabeledRecord(label.encoded, tuple(tuple(x) for x in targets), label.source)
        cache, digest = reference_cache(model, [hard_label], 0, device, cache_root, False)
        cached, restored_digest = reference_cache(model, [hard_label], 0, device, cache_root, True)
        assert cached == cache and restored_digest == digest
        # Simulate node-local storage with a separate directory per worker.
        # Each worker represents local rank zero of a different machine.
        node_cache = cache_root / f"node_{dist.get_rank()}"
        node_cache.mkdir()
        local_rank = os.environ["LOCAL_RANK"]
        os.environ["LOCAL_RANK"] = "0"
        node_reference, node_digest = reference_cache(model, [hard_label], 0, device, node_cache, False)
        node_restored, restored_node_digest = reference_cache(model, [hard_label], 0, device, node_cache, True)
        os.environ["LOCAL_RANK"] = local_rank
        assert node_reference == node_restored == cache and node_digest == restored_node_digest == digest
        (node_cache / "reference.json").unlink()
        node_cache.rmdir()
        batch = collate_records([label.encoded], pad_token_id=0, device=device)
        with torch.no_grad():
            fields = model(batch)[0]
            reference = [x.float().softmax(-1).tolist() for x in fields]
            group = sample_group(fields, targets, 32, generator)
        loss, _ = group_loss(model(batch)[0], targets, group, reference)
        loss.backward()
        assert all(torch.isfinite(p.grad.to_local()).all() for p in model.parameters() if p.grad is not None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
        optimizer.step()
        saved = {name: parameter.detach().to_local().clone() for name, parameter in model.named_parameters()}
        dcp.save({"model": ModelState(model), "optimizer": OptimizerState(model, optimizer)}, checkpoint_id=cache_root / "native")
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
        dcp.load({"model": ModelState(model), "optimizer": OptimizerState(model, optimizer)}, checkpoint_id=cache_root / "native")
        assert all(torch.equal(saved[name], parameter.detach().to_local()) for name, parameter in model.named_parameters())
        training_args = TrainArgs(underscores_to_dashes=True).parse_args(
            [
                "--model-dir",
                str(cache_root),
                "--data-dir",
                str(cache_root),
                "--output-dir",
                str(cache_root),
                "--run-name",
                "fixture-only",
                "--global-batch-size",
                "2",
            ]
        )
        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        trace = io.StringIO()
        rows, metrics = train_step(model, optimizer, [None, None], [hard_label, hard_label], cache, processor, 0, training_args, device, trace)
        assert len(rows) == 4 and metrics["rl/loss"] == metrics["rl/loss"] and '"rewards"' in trace.getvalue()
        dist.barrier()
        if dist.get_rank() == 0:
            (cache_root / "reference.json").unlink()
            # Native fixture artifacts remain for inspection; no model weights
            # from the production checkpoint are written by this test.
        dist.destroy_process_group()
    print("PASS: policy gradient, constant groups, clipping, KL, hard labels, real Qwen/head gradients, strict head reload, optional FSDP update")


if __name__ == "__main__":
    main()
