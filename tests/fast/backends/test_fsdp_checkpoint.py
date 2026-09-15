import importlib.util
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _load_checkpoint_module():
    # Avoid importing the FSDP actor and its GPU dependencies in CPU tests.
    path = Path(__file__).resolve().parents[3] / "miles/backends/fsdp_utils/checkpoint.py"
    spec = importlib.util.spec_from_file_location("fsdp_checkpoint_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_worker(rank, root, mode):
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        checkpoint = _load_checkpoint_module()
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(0.5))
        actor = SimpleNamespace(
            args=SimpleNamespace(save=root, load=root, no_save_optim=True, no_load_rng=mode == "disabled"),
            model=model,
            global_step=7,
            micro_step=14,
        )
        # The same initial seed can diverge after rank-dependent work.
        torch.rand(rank + 1)
        cuda_generator = torch.Generator().manual_seed(100 + rank)
        with patch.object(torch.cuda, "synchronize"), patch.object(
            torch.cuda, "get_rng_state_all", side_effect=lambda: [cuda_generator.get_state()]
        ):
            checkpoint.save(actor, iteration=6)

        expected_rng = torch.get_rng_state()
        expected_cuda_rng = cuda_generator.get_state()
        expected_output = model(torch.ones(4, 8))
        expected_output.sum().backward()
        expected_grad = model[0].weight.grad.clone()
        model.zero_grad()

        checkpoint_dir = Path(root) / "iter_0000007"
        if mode == "legacy":
            if rank == 0:
                (checkpoint_dir / "rng_rank_0.pt").rename(checkpoint_dir / "rng.pt")
                (checkpoint_dir / "rng_rank_1.pt").unlink()
            dist.barrier()
            legacy_rng = torch.load(checkpoint_dir / "rng.pt", weights_only=True)
            expected_rng = legacy_rng["torch"]
            expected_cuda_rng = legacy_rng["cuda"][0]

        torch.manual_seed(999)
        cuda_generator.manual_seed(999)
        if mode == "disabled":
            expected_rng = torch.get_rng_state()
            expected_cuda_rng = cuda_generator.get_state()
        with torch.no_grad():
            model[0].weight.zero_()
        payload = checkpoint.load(actor)
        assert payload is not None
        with patch.object(torch.cuda, "synchronize"), patch.object(
            torch.cuda, "is_available", return_value=True
        ), patch.object(
            torch.cuda, "set_rng_state_all", side_effect=lambda states: cuda_generator.set_state(states[0])
        ):
            checkpoint.finalize_load(actor, payload)

        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert torch.equal(cuda_generator.get_state(), expected_cuda_rng)
        assert actor.args.start_rollout_id == 7
        if mode == "rank":
            output = model(torch.ones(4, 8))
            output.sum().backward()
            assert torch.equal(output, expected_output)
            assert torch.equal(model[0].weight.grad, expected_grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["rank", "legacy", "disabled"])
def test_checkpoint_restores_rank_rng(tmp_path, mode):
    mp.spawn(_checkpoint_worker, args=(str(tmp_path), mode), nprocs=2, join=True)
