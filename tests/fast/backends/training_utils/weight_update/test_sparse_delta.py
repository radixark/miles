import argparse
import pathlib
from unittest import mock

import safetensors.torch
import torch
from sglang.srt import utils as sglang_utils

from miles.backends.training_utils.weight_update.protocols import sparse_delta as sparse_delta_protocol
from miles.utils import sparse_delta

_DELTA_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"
_SPARSE_MODULE = "miles.backends.training_utils.weight_update.protocols.sparse_delta"


class _Model(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)

    def load_weights(self, weights) -> None:
        for _, tensor in weights:
            self.weight.data.copy_(tensor)


class _Engine:
    """Stands in for SGLang's update_weights_from_tensor."""

    def __init__(self, model: _Model) -> None:
        self.model = model

    async def update_weights_from_tensor(self, *, serialized_named_tensors, load_format, selector) -> dict:
        assert load_format == sparse_delta.LOADER and selector == "all"
        request, same = serialized_named_tensors
        assert request == same
        sparse_delta.apply_delta(self.model, sglang_utils.MultiprocessingSerializer.deserialize(request))
        return {"success": True}


def test_engine_matches_trainer_after_each_sync(tmp_path: pathlib.Path) -> None:
    baseline = torch.tensor([0.5, -1.0, 2.0, 3.0], dtype=torch.bfloat16)
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    safetensors.torch.save_file({"weight": baseline}, checkpoint / "model.safetensors")
    protocol = sparse_delta_protocol.UpdateWeightFromSparseDelta(
        argparse.Namespace(
            hf_checkpoint=str(checkpoint),
            update_weight_disk_dir=str(tmp_path / "deltas"),
            update_weight_delta_encoding="sparse-xor",
            update_weight_delta_checksum="adler32",
            update_weight_delta_verify=True,
            custom_update_weight_post_write_path=None,
            sglang=argparse.Namespace(base_args={"custom_pull_weights_pre_read_hook": None}),
        )
    )
    engine = _Engine(_Model(baseline))
    protocol.rollout_engines = [engine]
    protocol.is_sender = True
    protocol._engine_gpu_counts = [2]

    with (
        mock.patch(f"{_DELTA_MODULE}.dist") as distributed,
        mock.patch(f"{_SPARSE_MODULE}.dist", distributed),
        mock.patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=None),
        mock.patch(f"{_SPARSE_MODULE}.distributed_utils.get_gloo_group", return_value=None),
        mock.patch.object(torch.cuda, "current_device", return_value="cpu"),
    ):
        distributed.get_rank.return_value = 0
        distributed.get_world_size.return_value = 1
        distributed.all_gather_object.side_effect = lambda output, value, **kwargs: output.__setitem__(0, value)
        for weight_version, first in enumerate([0.5, 0.75, 0.75, -0.0], start=1):
            weights = torch.tensor([first, -1.0, 2.0, 4.0 if weight_version > 2 else 3.0], dtype=torch.bfloat16)
            assert protocol.begin_sync(
                weight_version=weight_version,
                iter_buckets=lambda weights=weights, **kwargs: [[("weight", weights)]],
            )
            protocol.send_bucket([("weight", weights)])
            protocol.after_base_weights()
            protocol.finalize(weight_version=weight_version)
            assert torch.equal(sparse_delta.get_bits(engine.model.weight.data), sparse_delta.get_bits(weights))
