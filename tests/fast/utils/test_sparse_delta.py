import json
import pathlib

import pytest
import safetensors.numpy
import safetensors.torch
import torch

from miles.utils import disk_delta, sparse_delta

_CHECKSUM = "adler32"


class _PostLoad:
    """A post-load step on a class the model does not inherit, like GLM-5.3's mixin."""

    def post_load_weights(model) -> None:
        model.derived.copy_(model.plain.data.float() * 2)


class _StackingModel(torch.nn.Module):
    """Stacks `a` and `b` into one param, ties `c` under two names, derives a tensor from `c`."""

    def __init__(self) -> None:
        super().__init__()
        self.stacked = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16), requires_grad=False)
        self.plain = torch.nn.Parameter(torch.zeros(5, dtype=torch.bfloat16), requires_grad=False)
        self.tied = self.plain
        self.derived = torch.zeros(5)

    def post_load_weights(self) -> None:
        _PostLoad.post_load_weights(self)

    def load_weights(self, weights) -> None:
        for name, tensor in weights:
            if name == "c":
                self.plain.data.copy_(tensor)
            else:
                rows = slice(0, 2) if name == "a" else slice(2, 4)
                self.stacked.data[rows].copy_(tensor)
        _PostLoad.post_load_weights(self)


_SHAPES = {"a": (2, 3), "b": (2, 3), "c": (5,)}


def _make_weights(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    # Random bits, so NaN payloads and -0.0 are covered.
    return {
        name: torch.randint(-(2**15), 2**15, shape, generator=generator, dtype=torch.int16).view(torch.bfloat16)
        for name, shape in _SHAPES.items()
    }


def _publish_version(root: pathlib.Path, version: int, old: dict, new: dict, full: bool = False) -> None:
    deltas, specs = {}, {}
    for name, tensor in new.items():
        old_bits = sparse_delta.get_bits(old[name].clone()).reshape(-1)
        encoded = sparse_delta.encode_delta(sparse_delta.get_bits(tensor).reshape(-1), old_bits).numpy()
        if encoded.size:
            encoded = sparse_delta.compress_delta(encoded)
            deltas[name] = encoded
            specs[name] = json.dumps(
                {"dtype": "bfloat16", "shape": list(tensor.shape), "checksum": disk_delta.checksum(_CHECKSUM, encoded)}
            )
    directory = root / f"weight_v{version:06d}"
    directory.mkdir(parents=True)
    safetensors.numpy.save_file(deltas, directory / "model-00000-of-00001.safetensors", metadata=specs)
    index = {
        "metadata": {"delta_encoding": sparse_delta.ENCODING, "checksum_format": _CHECKSUM},
        "weight_map": {name: "model-00000-of-00001.safetensors" for name in deltas},
    }
    (directory / sparse_delta.INDEX).write_text(json.dumps(index))
    if full:
        safetensors.torch.save_file(new, directory / sparse_delta.FULL_FILES.replace("*", "00000"))


def _get_model_bits(model: _StackingModel) -> list[torch.Tensor]:
    return [sparse_delta.get_bits(weight.data).clone() for weight in model.parameters()]


def test_applied_deltas_equal_a_full_load(tmp_path: pathlib.Path) -> None:
    base, middle, last = _make_weights(0), _make_weights(1), _make_weights(2)
    middle["b"] = base["b"]
    _publish_version(tmp_path, 1, base, middle)
    _publish_version(tmp_path, 2, middle, last)
    model, expected = _StackingModel(), _StackingModel()
    model.load_weights(base.items())
    expected.load_weights(last.items())

    sparse_delta.apply_delta(model, sparse_delta.build_request(str(tmp_path), 2, None, [], False))

    for got, want in zip(_get_model_bits(model), _get_model_bits(expected), strict=True):
        assert torch.equal(got, want)
    assert torch.equal(model.derived, expected.derived)
    with pytest.raises(ValueError, match="holds version 2"):
        sparse_delta.apply_delta(model, sparse_delta.build_request(str(tmp_path), 2, None, [], False))


def test_a_corrupted_delta_is_refused(tmp_path: pathlib.Path) -> None:
    base, new = _make_weights(0), _make_weights(1)
    _publish_version(tmp_path, 1, base, new)
    path = tmp_path / "weight_v000001" / "model-00000-of-00001.safetensors"
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 1
    path.write_bytes(blob)
    model = _StackingModel()
    model.load_weights(base.items())

    with pytest.raises(ValueError, match="checksum mismatch"):
        sparse_delta.apply_delta(model, sparse_delta.build_request(str(tmp_path), 1, None, [], False))


class _ScalingModel(_StackingModel):
    """A loader that transforms values."""

    def load_weights(self, weights) -> None:
        super().load_weights((name, tensor * 2 if name == "c" else tensor) for name, tensor in weights)


def test_verify_rejects_a_transforming_loader(tmp_path: pathlib.Path) -> None:
    generator = torch.Generator().manual_seed(0)
    base = {name: torch.rand(shape, generator=generator).to(torch.bfloat16) for name, shape in _SHAPES.items()}
    new = {name: tensor + 0.25 for name, tensor in base.items()}
    _publish_version(tmp_path, 1, base, new, full=True)
    request = sparse_delta.build_request(str(tmp_path), 1, None, [], True)
    copying, scaling = _StackingModel(), _ScalingModel()
    copying.load_weights(base.items())
    scaling.load_weights(base.items())

    sparse_delta.apply_delta(copying, request)

    with pytest.raises(ValueError, match="differ between the delta and a full load"):
        sparse_delta.apply_delta(scaling, request)
