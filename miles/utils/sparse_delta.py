"""Sparse XOR weight deltas: encoded on the trainer's GPU, applied in place on the engine.

A tensor's delta is the gaps between its changed elements, then old XOR new for each, as byte planes under zstd.
`apply_delta` is an SGLang --custom-weight-loader: it zeroes the weights, loads the delta through the model's own
`load_weights` and XORs the old weights back. Only correct for loaders that copy values unchanged.
"""

import base64
import collections
import contextlib
import importlib
import json
import logging
import math
import pathlib
import sys
import time
import typing
import weakref
from concurrent import futures

import numpy as np
import safetensors.numpy
import safetensors.torch
import torch
import zstandard

from miles.backends.training_utils.weight_update.protocols.utils import loader_probe
from miles.utils import disk_delta

logger = logging.getLogger(__name__)

INDEX = "model.safetensors.index.json"
ENCODING = "sparse-xor"
LOADER = "miles.utils.sparse_delta.apply_delta"
# Full weights published next to the delta when verifying.
FULL_FILES = "full-*.safetensors"

_INT_DTYPES = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
# Applied version per model; 0 is the base checkpoint.
_VERSIONS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
# Per model: params written per HF tensor, and written buffers.
_LOADER_WRITES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class _ParamLayout(typing.NamedTuple):
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype


def get_bits(tensor: torch.Tensor) -> torch.Tensor:
    """View the tensor as integers of the same width."""
    return tensor.view(_INT_DTYPES[tensor.element_size()])


def encode_delta(new_bits: torch.Tensor, old_bits: torch.Tensor) -> torch.Tensor:
    """Encode flat `old_bits` -> `new_bits` as byte planes; updates `old_bits` in place."""
    if new_bits.numel() >= 2**31:
        raise ValueError(f"a tensor of {new_bits.numel()} elements does not fit the delta's int32 gaps")
    index = torch.nonzero(new_bits != old_bits).flatten()
    new = new_bits[index]
    xor = new ^ old_bits[index]
    old_bits[index] = new
    gaps = torch.diff(index, prepend=index.new_zeros(1)).to(torch.int32)
    return torch.cat([_split_planes(gaps), _split_planes(xor)])


def decode_delta(delta: torch.Tensor, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    """Dense XOR tensor of an encoded delta."""
    count = delta.numel() // (4 + dtype.itemsize)
    bits = torch.zeros(math.prod(shape), dtype=_INT_DTYPES[dtype.itemsize], device=delta.device)
    index = torch.cumsum(_join_planes(delta[: 4 * count], torch.int32), dim=0)
    bits[index] = _join_planes(delta[4 * count :], bits.dtype)
    return bits.view(dtype).reshape(shape)


def compress_delta(delta: np.ndarray) -> np.ndarray:
    return np.frombuffer(zstandard.ZstdCompressor(level=1).compress(delta), dtype=np.uint8)


def _split_planes(values: torch.Tensor) -> torch.Tensor:
    """Regroup the values' bytes by significance."""
    return values.view(torch.uint8).view(-1, values.element_size()).T.reshape(-1)


def _join_planes(planes: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # view(dtype) needs the contiguous copy.
    return planes.view(dtype.itemsize, -1).T.reshape(-1).clone().view(dtype)


def build_request(
    source_dir: str, version: int, pre_read_hook: str | None, sources: list[str], verify: bool
) -> list[tuple[str, torch.Tensor]]:
    """Named tensors that carry an `apply_delta` request through update_weights_from_tensor."""
    request = {
        "source_dir": source_dir,
        "version": version,
        "pre_read_hook": pre_read_hook,
        "sources": sources,
        "verify": verify,
    }
    # The request goes in the name: SGLang ships a non-empty CPU tensor by fd, and a colon marks LoRA.
    return [(base64.urlsafe_b64encode(json.dumps(request).encode()).decode(), torch.empty(0))]


def apply_delta(model: torch.nn.Module, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
    """Bring `model` to the requested version."""
    ((request_name, _),) = named_tensors
    request = json.loads(base64.urlsafe_b64decode(request_name))
    if request["pre_read_hook"]:
        module_path, _, name = request["pre_read_hook"].rpartition(".")
        hook = getattr(importlib.import_module(module_path), name)  # config-access-exempt: hook named by the request
        hook(request["source_dir"], request["version"], *([request["sources"]] if request["sources"] else []))
    held = _VERSIONS.get(model, 0)
    if request["version"] <= held:
        raise ValueError(f"the engine holds version {held}; version {request['version']} would apply nothing")
    with _defer_post_load(model):
        for version in range(held + 1, request["version"] + 1):
            _apply_version(model, pathlib.Path(request["source_dir"]) / f"weight_v{version:06d}", request["verify"])
            _VERSIONS[model] = version


@contextlib.contextmanager
def _defer_post_load(model: torch.nn.Module):
    """Run `post_load_weights` once after the block, not after every `load_weights` in it."""
    # A loader may call another class's step (GLM-5.3 calls a mixin's), so patch the whole package.
    package = type(model).__module__.partition(".")[0]
    modules = [module for name, module in list(sys.modules.items()) if f"{name}.".startswith(f"{package}.")]
    classes = {id(value): value for module in modules for value in vars(module).values() if isinstance(value, type)}
    steps = {cls: vars(cls)["post_load_weights"] for cls in classes.values() if "post_load_weights" in vars(cls)}
    for cls in steps:
        cls.post_load_weights = lambda *args, **kwargs: None
    try:
        yield
    finally:
        for cls, step in steps.items():
            cls.post_load_weights = step
    if hasattr(model, "post_load_weights"):  # config-access-exempt: optional inference-model post-load step
        model.post_load_weights()


def _apply_version(model: torch.nn.Module, version_dir: pathlib.Path, verify: bool) -> None:
    index = json.loads((version_dir / INDEX).read_text())
    metadata = index["metadata"]
    if metadata["delta_encoding"] != ENCODING:
        raise ValueError(f"{version_dir} is a {metadata['delta_encoding']!r} delta, not {ENCODING!r}")
    start = time.monotonic()
    device = next(model.parameters()).device
    deltas: dict[str, np.ndarray] = {}
    specs: dict[str, dict] = {}
    for filename in sorted(set(index["weight_map"].values())):
        path = version_dir / filename
        deltas.update(safetensors.numpy.load_file(path))
        specs.update(
            (name, json.loads(spec)) for name, spec in safetensors.safe_open(path, framework="np").metadata().items()
        )

    def decode(name: str) -> tuple[str, torch.Tensor]:
        if disk_delta.checksum(metadata["checksum_format"], deltas[name]) != specs[name]["checksum"]:
            raise ValueError(f"checksum mismatch for {name} in {version_dir}")
        delta = np.frombuffer(zstandard.ZstdDecompressor().decompress(deltas[name]), dtype=np.uint8)
        dtype = getattr(torch, specs[name]["dtype"])  # config-access-exempt: dtype named by the delta file
        return name, decode_delta(torch.tensor(delta, device=device), dtype, tuple(specs[name]["shape"]))

    hf_specs = {
        name: loader_probe.HfTensorSpec(
            tuple(spec["shape"]), getattr(torch, spec["dtype"])  # config-access-exempt: dtype named by the delta file
        )
        for name, spec in specs.items()
    }
    buffers = _get_written_buffers(model, hf_specs, device)
    old_buffers = [buffer.detach().clone() for buffer in buffers]
    for buffer in buffers:
        buffer.zero_()
    landed = 0
    with futures.ThreadPoolExecutor(disk_delta.NUM_WORKERS) as pool:
        for names, weights in _iter_groups(model, hf_specs, device):
            old = [weight.detach().clone() for weight in weights]
            for weight in weights:
                weight.data.zero_()
            model.load_weights(pool.map(decode, names))
            landed += sum(int(torch.count_nonzero(get_bits(weight.data))) for weight in weights)
            for weight, before in zip(weights, old, strict=True):
                get_bits(weight.data).bitwise_xor_(get_bits(before))
    for buffer, before in zip(buffers, old_buffers, strict=True):
        get_bits(buffer).bitwise_xor_(get_bits(before))
    logger.info("applied %s in %.2f s: %d changed elements landed", version_dir.name, time.monotonic() - start, landed)
    if verify:
        _verify_against_full_load(model, version_dir, device)


def _verify_against_full_load(model: torch.nn.Module, version_dir: pathlib.Path, device: torch.device) -> None:
    """Raise unless a full load of the version leaves the same bits."""
    differing = 0
    for path in sorted(version_dir.glob(FULL_FILES)):
        full = safetensors.torch.load_file(path)
        hf_specs = {
            name: loader_probe.HfTensorSpec(tuple(tensor.shape), tensor.dtype) for name, tensor in full.items()
        }
        for names, weights in _iter_groups(model, hf_specs, device):
            applied = [weight.detach().clone() for weight in weights]
            model.load_weights((name, full[name].to(device)) for name in names)
            differing += sum(
                int(torch.count_nonzero(get_bits(weight.data) ^ get_bits(before)))
                for weight, before in zip(weights, applied, strict=True)
            )
    if differing:
        raise ValueError(f"{version_dir.name}: {differing} elements differ between the delta and a full load")
    logger.info("verified %s: a full load leaves the same weights", version_dir.name)


def _iter_groups(model: torch.nn.Module, hf_specs: dict[str, loader_probe.HfTensorSpec], device: torch.device):
    """Yield (HF names, params) for names whose loaders write the same params."""
    writes, _ = _get_loader_writes(model, hf_specs, device)
    roots: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node: tuple[str, str]) -> tuple[str, str]:
        while roots.setdefault(node, node) != node:
            roots[node] = node = roots[roots[node]]
        return node

    for name in hf_specs:
        for param_name in writes[name]:
            roots[find(("hf", name))] = find(("param", param_name))
    groups: dict[tuple[str, str], tuple[list[str], set[str]]] = collections.defaultdict(lambda: ([], set()))
    for name in hf_specs:
        names, param_names = groups[find(("hf", name))]
        names.append(name)
        param_names.update(writes[name])
    params = dict(model.named_parameters())
    for names, param_names in groups.values():
        if param_names:
            # Dedupe tied weights: a second XOR would undo the first.
            yield names, list({params[name].data_ptr(): params[name] for name in sorted(param_names)}.values())


def _get_written_buffers(
    model: torch.nn.Module, hf_specs: dict[str, loader_probe.HfTensorSpec], device: torch.device
) -> list[torch.Tensor]:
    _, buffer_names = _get_loader_writes(model, hf_specs, device)
    buffers = dict(model.named_buffers())
    return [buffers[name] for name in sorted(buffer_names)]


def _get_loader_writes(
    model: torch.nn.Module, hf_specs: dict[str, loader_probe.HfTensorSpec], device: torch.device
) -> tuple[dict[str, frozenset[str]], set[str]]:
    """Params and buffers the model's loader writes per HF tensor, probed once per name."""
    writes, buffer_names = _LOADER_WRITES.setdefault(model, ({}, set()))
    unknown = {name: spec for name, spec in hf_specs.items() if name not in writes}
    if unknown:
        layouts = {
            name: _ParamLayout(param.shape, param.stride(), param.dtype) for name, param in model.named_parameters()
        }
        # The probe writes the model's real buffers; restore them.
        saved = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
        probed = loader_probe.probe_loader_writes(model, layouts, unknown, device=device)
        for buffer, before in saved:
            buffer.copy_(before)
        for name in unknown:
            writes[name] = probed.hf_name_mapping.param_names_by_hf_name.get(name, frozenset())
        buffer_names.update(probed.buffer_names)
    return writes, buffer_names
