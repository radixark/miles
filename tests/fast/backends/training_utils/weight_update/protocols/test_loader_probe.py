"""`probe_loader_writes` must report exactly what a model's own loader writes: params by HF name, and buffers."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.utils.loader_probe import HfTensorSpec, probe_loader_writes

_LOCAL_EXPERT_IDS = (2, 3)
_CPU = torch.device("cpu")


def _empty_param(*shape: int) -> torch.nn.Parameter:
    # as a model replica keeps it: 0-size, with its loader as an attribute
    param = torch.nn.Parameter(torch.empty(0), requires_grad=False)
    param.built_shape = shape
    return param


class _ToyModel(torch.nn.Module):
    """Exercise fused, sharded and shared-input loads, scalar reads and derived buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = _empty_param(12)
        self.w13 = _empty_param(len(_LOCAL_EXPERT_IDS), 4)
        self.fused_a = _empty_param(6)
        self.norm = _empty_param(4)
        self.scale = _empty_param()
        self.embed = _empty_param(6, 2)
        self.lm_head = _empty_param(6, 2)
        self.qkv.weight_loader = self._load_qkv_shard
        # buffers stay on meta, as a model replica keeps them
        self.register_buffer("norm_plus_one", torch.empty(4, device="meta"), persistent=False)
        self.register_buffer("cos_sin_cache", torch.empty(2, device="meta"), persistent=False)

    def _load_qkv_shard(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: str) -> None:
        offset = {"q": 0, "k": 4, "v": 8}[shard_id]
        param.data.narrow(0, offset, 4).copy_(loaded_weight)

    def load_weights(self, weights) -> None:
        cached_a_proj = {}
        for name, loaded_weight in weights:
            if name in ("q", "k", "v"):
                self.qkv.weight_loader(self.qkv, loaded_weight, name)
            elif name.startswith("experts."):
                expert_id = int(name.split(".")[1])
                if expert_id not in _LOCAL_EXPERT_IDS:
                    continue
                self.w13.data[_LOCAL_EXPERT_IDS.index(expert_id)].copy_(loaded_weight)
            elif name in ("q_a", "kv_a"):
                cached_a_proj[name] = loaded_weight
                if len(cached_a_proj) == 2:
                    self.fused_a.data.copy_(torch.cat([cached_a_proj["q_a"], cached_a_proj["kv_a"]]))
            elif name == "norm":
                self.norm.data[:] = loaded_weight
                torch.add(self.norm.data, 1.0, out=self.norm_plus_one)
            elif name == "scale":
                self.scale.data.fill_(loaded_weight.item())
            elif name == "embed":
                for param in (self.embed, self.lm_head):
                    param.data[:5].copy_(loaded_weight)
                    param.data[5:].fill_(0)


_HF_TENSOR_SPECS = {
    "q": HfTensorSpec((4,), torch.float32),
    "k": HfTensorSpec((4,), torch.float32),
    "v": HfTensorSpec((4,), torch.float32),
    **{f"experts.{expert_id}.w": HfTensorSpec((4,), torch.float32) for expert_id in range(4)},
    "q_a": HfTensorSpec((2,), torch.float32),
    "kv_a": HfTensorSpec((4,), torch.float32),
    "norm": HfTensorSpec((4,), torch.float32),
    "scale": HfTensorSpec((), torch.float32),
    "embed": HfTensorSpec((5, 2), torch.float32),
    "rotary.inv_freq": HfTensorSpec((2,), torch.float32),
}


def _param_layouts(model: torch.nn.Module) -> dict[str, SimpleNamespace]:
    return {
        name: SimpleNamespace(
            shape=param.built_shape,
            stride=torch.empty(param.built_shape).stride(),
            dtype=param.dtype,
        )
        for name, param in model.named_parameters()
    }


def test_each_hf_name_maps_to_the_params_its_loader_writes() -> None:
    """Fused and shared inputs must retain their dependencies; ignored inputs and padding must not add any."""
    model = _ToyModel()

    loader_writes = probe_loader_writes(model, _param_layouts(model), _HF_TENSOR_SPECS, device=_CPU)
    mapping = loader_writes.hf_name_mapping

    assert mapping.hf_names_by_param_name == {
        "qkv": {"q", "k", "v"},
        "w13": {"experts.2.w", "experts.3.w"},
        "fused_a": {"q_a", "kv_a"},
        "norm": {"norm"},
        "scale": {"scale"},
        "embed": {"embed"},
        "lm_head": {"embed"},
    }
    assert mapping.param_names_by_hf_name["embed"] == {"embed", "lm_head"}
    assert "experts.0.w" not in mapping.param_names_by_hf_name
    assert "rotary.inv_freq" not in mapping.param_names_by_hf_name
    assert loader_writes.buffer_names == {"norm_plus_one"}


def test_params_come_back_as_they_were() -> None:
    """The replica keeps loading into its own 0-size params with their attributes after the probe."""
    model = _ToyModel()
    params_before = dict(model.named_parameters())

    probe_loader_writes(model, _param_layouts(model), _HF_TENSOR_SPECS, device=_CPU)

    for name, param in model.named_parameters():
        assert param is params_before[name]
        assert param.numel() == 0 and param.device.type == "cpu"
    assert model.qkv.weight_loader == model._load_qkv_shard


def test_a_param_registered_under_two_names_is_reported_once() -> None:
    """`named_parameters()` lists a shared param once, and the replica lays it out under that name."""
    model = _ToyModel()
    model.alias = torch.nn.Module()
    model.alias.norm = model.norm

    mapping = probe_loader_writes(model, _param_layouts(model), _HF_TENSOR_SPECS, device=_CPU).hf_name_mapping

    assert mapping.param_names_by_hf_name["norm"] == {"norm"}


def test_a_loader_error_propagates_and_params_come_back() -> None:
    model = _ToyModel()
    params_before = dict(model.named_parameters())

    with pytest.raises(RuntimeError):
        probe_loader_writes(
            model, _param_layouts(model), {**_HF_TENSOR_SPECS, "q": HfTensorSpec((5,), torch.float32)}, device=_CPU
        )

    assert all(param is params_before[name] for name, param in model.named_parameters())


class _PackedScaleModel(torch.nn.Module):
    """Require repeated scale rows and a repacking kernel with no meta implementation."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = _empty_param(2)

    def load_weights(self, weights) -> None:
        for _, loaded_weight in weights:
            if not torch.all(loaded_weight == loaded_weight[0]):
                raise AssertionError("repeated scale rows differ")
            self.scale.data.copy_(_repack(loaded_weight)[0])


def _repack(tensor: torch.Tensor) -> torch.Tensor:
    # DeepGEMM's DLPack path requires real storage
    if tensor.is_meta:
        raise RuntimeError("Cannot pack tensors on meta")
    return tensor.clone()


def test_the_loader_sees_values_on_a_device_as_in_a_real_load() -> None:
    """Meta inputs cannot satisfy UE8M0 value checks or DeepGEMM's storage requirement."""
    model = _PackedScaleModel()

    mapping = probe_loader_writes(
        model, _param_layouts(model), {"scale": HfTensorSpec((3, 2), torch.int32)}, device=_CPU
    ).hf_name_mapping

    assert mapping.hf_names_by_param_name == {"scale": {"scale"}}
