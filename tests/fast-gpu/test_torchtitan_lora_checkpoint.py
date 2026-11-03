from tests.ci.ci_register import register_cuda_ci

# CPU-only checks, on this stage because only the GPU image ships torchtitan
register_cuda_ci(
    est_time=30, suite="stage-b-2-gpu-h200", labels=["torchtitan", "lora"], hardware=["hopper"], num_gpus=2
)

import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

# the package applies the torch 2.14 backports torchtitan needs at import, so it must load first
import miles.backends.torchtitan_utils  # noqa: F401

# isort: split
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3 import build_model_config

from miles.backends.torchtitan_utils.lora_checkpoint import probe_torchtitan_row_mapping
from miles.backends.torchtitan_utils.lora_names import iter_hf_adapter_tensors
from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    map_named_tensors_to_hf_keys,
    map_peft_keys_to_hf_keys,
)
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

_TARGETS = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 28})
_QKV, _WO, _W13 = "layers.3.attention.qkv_linear.wqkv", "layers.3.attention.wo", "layers.3.feed_forward.w13"


def _adapter(rank: int = 8) -> tuple[object, dict[str, torch.Tensor]]:
    model = build_model_config("0.6B", parallelism=ParallelismConfig(), seq_len=4096, attn_backend="flex")
    sd_adapter = model._owner.state_dict_adapter_cls(model, None)
    qkv = model.layers[3].attention.qkv_linear.wqkv
    g = torch.Generator().manual_seed(0)
    params = {
        f"{_QKV}.lora_a.weight": torch.randn(rank, qkv.in_features, generator=g),
        f"{_QKV}.lora_b.weight": torch.randn(qkv.out_features, rank, generator=g),
        f"{_WO}.lora_a.weight": torch.randn(rank, 2048, generator=g),
        f"{_WO}.lora_b.weight": torch.randn(1024, rank, generator=g),
        f"{_W13}.lora_a.weight": torch.randn(rank, 1024, generator=g),
        f"{_W13}.lora_b.weight": torch.randn(2, 3072, rank, generator=g),
    }
    return sd_adapter, params


def _expected(sd_adapter, params) -> dict[str, torch.Tensor]:
    weights, _ = map_named_tensors_to_hf_keys(
        iter_hf_adapter_tensors(sd_adapter, params), map_peft_keys_to_hf_keys, _TARGETS
    )
    return weights


def _check(mapping, params_local: list[torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    flat = torch.cat([p.reshape(-1) for p in params_local])
    for piece in mapping.pieces:
        box = tuple(slice(o, o + n) for o, n in zip(piece.offsets, piece.local_elements.shape, strict=True))
        assert torch.equal(flat[piece.local_elements], expected[piece.key][box]), piece.key


def _whole(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    sd_adapter, params = _adapter()
    mapping = probe_torchtitan_row_mapping(params, sd_adapter, _TARGETS)
    attn, mlp = "model.layers.3.self_attn", "model.layers.3.mlp"
    assert mapping.shared_lora_a_groups == (
        (f"{mlp}.gate_proj", f"{mlp}.up_proj"),
        (f"{attn}.q_proj", f"{attn}.k_proj", f"{attn}.v_proj"),
    )
    # each shared lora_A is stored once, under its group's first member
    assert sorted(p.key for p in mapping.pieces) == sorted(
        [f"{attn}.q_proj.lora_A", f"{attn}.o_proj.lora_A", f"{mlp}.gate_proj.lora_A"]
        + [f"{attn}.{m}_proj.lora_B" for m in "qkvo"]
        + [f"{mlp}.{m}_proj.lora_B" for m in ("gate", "up")]
    )
    _check(mapping, list(params.values()), _expected(sd_adapter, params))
    dist.destroy_process_group()


def test_whole_tensors_map_through_the_models_own_split():
    run_multiprocess(_whole, world_size=1)


def _sharded(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    mesh = init_device_mesh("cpu", (world_size,))
    sd_adapter, params = _adapter()
    expected = _expected(sd_adapter, params)
    sharded = {key: distribute_tensor(value, mesh, [Shard(0)]) for key, value in params.items()}
    mapping = probe_torchtitan_row_mapping(sharded, sd_adapter, _TARGETS)
    _check(mapping, [p.to_local() for p in sharded.values()], expected)
    dist.destroy_process_group()


def test_fsdp_shards_map_to_rectangles_of_the_hf_tensors():
    run_multiprocess(_sharded)
