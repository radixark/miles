from __future__ import annotations

import contextlib
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="stage-b-2-gpu-h200", labels=["megatron"])

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def host():
    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear
    from megatron.core import parallel_state as mpu
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer.optimizer_config import OptimizerConfig
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.transformer_config import TransformerConfig

    from miles.backends.megatron_utils.actor import MegatronTrainRayActor
    from miles.backends.megatron_utils.multi_lora_optimizer import build_multi_lora_optimizer
    from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

    from miles.utils.distributed_utils import init_gloo_group

    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{_free_port()}", rank=0, world_size=1)
    init_gloo_group()
    torch.cuda.set_device(0)
    mpu.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1234)
    trivial = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=trivial,
            cp=trivial,
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
            is_pp_last_stage=True,
        )
    )
    tf_config = TransformerConfig(
        num_layers=1, hidden_size=16, num_attention_heads=1, bf16=True, params_dtype=torch.bfloat16
    )
    base = ColumnParallelLinear(
        16, 16, config=tf_config, init_method=torch.nn.init.xavier_uniform_, bias=False, gather_output=False
    )
    layer = MultiLoRALinear(base.cuda().bfloat16(), n_adapters=2, dim=4, alpha=8.0, full_name="linear_fc1")

    class Toy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.layer = inner

        def forward(self, x):
            return self.layer(x)[0]

    module = Toy(layer).cuda().bfloat16()
    for name, param in module.named_parameters():
        param.requires_grad = ".adapters." in name
    ddp = DistributedDataParallel(
        tf_config, DistributedDataParallelConfig(grad_reduce_in_fp32=True, use_distributed_optimizer=False), module
    )
    args = SimpleNamespace(
        multi_lora_n_adapters=2,
        use_gloo_process_groups=False,
        lora_rank=4,
        target_modules=None,
        hf_checkpoint=None,
        lora_dropout=0.0,
    )
    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        bf16=True,
        fp16=False,
        use_distributed_optimizer=False,
        clip_grad=0.0,
    )
    optimizer = build_multi_lora_optimizer(args, opt_config, [ddp])
    handler_names = [name for name in dir(MegatronTrainRayActor) if name.startswith(("multi_lora_", "_multi_lora_"))]
    holder = type("Host", (), {name: getattr(MegatronTrainRayActor, name) for name in handler_names})()
    holder.args, holder.model, holder.optimizer, holder._multi_lora_bindings = args, [ddp], optimizer, {}
    yield holder, ddp
    dist.destroy_process_group()


def test_tinker_operations_map_to_trainer_handlers(host, tmp_path):
    from megatron.bridge.peft.multi_lora_layers import set_tokens_per_adapter_slot

    from miles.backends.megatron_utils.multi_lora_optimizer import (
        adapter_slot_parameters,
        reset_grad_metadata_keep_grads,
    )
    from miles.utils.multi_lora import AdapterIdentity

    holder, ddp = host
    ident = AdapterIdentity(name="a", registration_id="r1", slot=0)
    other = AdapterIdentity(name="b", registration_id="r2", slot=1)

    assert holder.multi_lora_create_model(ident, adapter_rank=4, alpha=8.0, seed=11) == {"loaded_tensors": 0}
    holder.multi_lora_create_model(other, adapter_rank=2, alpha=8.0)
    with pytest.raises(ValueError):
        holder.multi_lora_create_model(ident, adapter_rank=4, alpha=8.0)

    fwd_out = {"log_probs": [torch.ones(2, device="cuda")], "request_loss": torch.tensor(1.0)}
    with (
        mock.patch("miles.backends.megatron_utils.actor.get_data_iterator", return_value=([], [1])),
        mock.patch("miles.backends.megatron_utils.model.forward_with_request_loss", return_value=fwd_out) as fwd,
    ):
        out = holder.multi_lora_forward(ident, rollout_data={}, request_loss_fn="cross_entropy")
    assert fwd.called and out["log_probs"][0].device.type == "cpu"

    with (
        mock.patch("miles.backends.megatron_utils.actor.get_data_iterator", return_value=([], [1])),
        mock.patch(
            "miles.backends.megatron_utils.model.forward_backward_with_request_loss",
            return_value=(SimpleNamespace(name="NORMAL"), [{"loss": 1.0}]),
        ) as fb,
    ):
        out = holder.multi_lora_forward_backward(ident, rollout_data={}, request_loss_fn="ppo")
    assert fb.called and out == {"outcome": "NORMAL", "losses": [{"loss": 1.0}]}
    with pytest.raises(ValueError):
        holder.multi_lora_forward(AdapterIdentity("x", "r9", 0), rollout_data={}, request_loss_fn="cross_entropy")

    reset_grad_metadata_keep_grads([ddp])
    set_tokens_per_adapter_slot([ddp], torch.tensor([4, 0], dtype=torch.int32, device="cuda"))
    x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
    ddp(x).float().square().sum().backward()
    ddp.finish_grad_sync()
    adam = {"lr": 1e-3, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.0}
    result = holder.multi_lora_optim_step(ident, grad_scale=1.0, clip_grad=0.0, adam_params=adam)
    assert result["grad_norm"] is not None and torch.isfinite(torch.tensor(result["grad_norm"]))
    assert all(p.main_grad.abs().max().item() == 0.0 for p in adapter_slot_parameters(ddp, 0))

    holder.multi_lora_clear_gradients(ident)

    fake_bridge = SimpleNamespace(
        export_adapter_weights=lambda model, cpu, show_progress: iter([("m.lora_A.weight", torch.zeros(4, 3), "n")])
    )
    auto_bridge = SimpleNamespace(from_hf_pretrained=lambda *a, **k: fake_bridge)
    with (
        mock.patch("megatron.bridge.AutoBridge", auto_bridge),
        mock.patch("miles.utils.megatron_bridge_utils.patch_megatron_model", lambda m: contextlib.nullcontext()),
    ):
        exported = holder.multi_lora_export_adapter(ident, adapter_rank=4)
        saved = holder.multi_lora_save_state(ident, save_dir=tmp_path, step=3, adapter_rank=4, alpha=8.0)
    assert "m.lora_A.weight" in exported
    ckpt = Path(saved["path"])
    assert (ckpt / "training_state_rank0.pt").exists() and (ckpt / "adapter_megatron_tp0_pp0.pt").exists()

    master = adapter_slot_parameters(ddp, 0)[0].main_param
    before = master.detach().clone()
    master.data.add_(1.0)
    holder.multi_lora_load_state(ident, ckpt_dir=ckpt, with_optimizer=True)
    assert torch.equal(master.detach(), before)
    holder.multi_lora_load_state(ident, ckpt_dir=ckpt, with_optimizer=False)

    holder.multi_lora_release_adapter(other)
    assert 1 not in holder._multi_lora_bindings
