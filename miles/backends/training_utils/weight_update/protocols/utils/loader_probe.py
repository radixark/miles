from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.weak import WeakIdKeyDictionary


class HfTensorSpec(NamedTuple):
    """An HF tensor the trainer sends, without its data."""

    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class HfNameMapping:
    """Which params of a model replica each HF name loads into, as the replica's own loader writes them, and the
    reverse. An HF name its loader ignores maps to no params."""

    param_names_by_hf_name: Mapping[str, frozenset[str]]
    hf_names_by_param_name: Mapping[str, frozenset[str]]

    @classmethod
    def from_hf_names_by_param_name(cls, hf_names_by_param_name: Mapping[str, frozenset[str]]) -> "HfNameMapping":
        param_names_by_hf_name = defaultdict(set)
        for param_name, hf_names in hf_names_by_param_name.items():
            for hf_name in hf_names:
                param_names_by_hf_name[hf_name].add(param_name)
        return cls(
            param_names_by_hf_name={name: frozenset(params) for name, params in param_names_by_hf_name.items()},
            hf_names_by_param_name=dict(hf_names_by_param_name),
        )

    def union(self, other: "HfNameMapping") -> "HfNameMapping":
        param_names = self.hf_names_by_param_name.keys() | other.hf_names_by_param_name.keys()
        return HfNameMapping.from_hf_names_by_param_name(
            {
                param_name: self.hf_names_by_param_name.get(param_name, frozenset())
                | other.hf_names_by_param_name.get(param_name, frozenset())
                for param_name in param_names
            }
        )


class LoaderWrites(NamedTuple):
    """What a model replica's loader writes: its params, by the HF names whose data lands in them, and the buffers
    it fills besides (sglang's Gemma norm derives `weight + 1` into one)."""

    hf_name_mapping: HfNameMapping
    buffer_names: frozenset[str]


def probe_loader_writes(
    model: torch.nn.Module,
    param_layouts: Mapping[str, object],
    hf_tensor_specs: Mapping[str, HfTensorSpec],
    *,
    device: torch.device,
) -> LoaderWrites:
    """Runs `model.load_weights` once over HF tensors of `hf_tensor_specs` and returns what it writes.

    The model replica calls it inside its rank's parallelism context, so the loader shards, places experts and fuses
    names as the engine's loader does. Params are swapped for meta ones in `param_layouts` (shape, stride, dtype)
    for the call and restored after it, so the probe allocates no param storage. The HF tensors are real, made one
    at a time on `device`, so the loader's checks and kernels see values on a device as in a real load.
    """
    originals, param_names_by_storage = _swap_params_for_meta(model, param_layouts)
    recorder = _SourceRecordingMode(param_names_by_storage, _get_buffer_names_by_storage(model))

    def iter_probe_tensors():
        for hf_name, spec in hf_tensor_specs.items():
            # ones pass the loaders' format checks: positive scales, equal repeated scale rows
            tensor = torch.ones(spec.shape, dtype=spec.dtype, device=device)
            recorder.tag(tensor, frozenset({hf_name}))
            yield hf_name, tensor

    try:
        with recorder:
            model.load_weights(iter_probe_tensors())
    finally:
        for module, local_name, param in originals:
            module._parameters[local_name] = param
    return LoaderWrites(
        hf_name_mapping=HfNameMapping.from_hf_names_by_param_name(
            {param_name: frozenset(hf_names) for param_name, hf_names in recorder.hf_names_by_param_name.items()}
        ),
        buffer_names=frozenset(recorder.written_buffer_names),
    )


def _swap_params_for_meta(
    model: torch.nn.Module, param_layouts: Mapping[str, object]
) -> tuple[list[tuple[torch.nn.Module, str, torch.nn.Parameter]], dict[StorageWeakRef, list[str]]]:
    # by the original's id, so a param shared by several modules stays shared; a meta tensor refuses `.data = <cpu>`
    param_names_by_param_id = defaultdict(list)
    for name, param in model.named_parameters(remove_duplicate=False):
        param_names_by_param_id[id(param)].append(name)
    replacements, originals = {}, []
    for module in model.modules():
        for local_name, param in list(module._parameters.items()):
            if param is None:
                continue
            if id(param) not in replacements:
                layout = param_layouts[param_names_by_param_id[id(param)][0]]
                meta_data = torch.empty_strided(layout.shape, layout.stride, dtype=layout.dtype, device="meta")
                replacement = torch.Tensor._make_subclass(type(param), meta_data, param.requires_grad)
                replacement.__dict__.update(param.__dict__)
                replacements[id(param)] = replacement
            originals.append((module, local_name, param))
            module._parameters[local_name] = replacements[id(param)]
    # only the name params are known by, so a param registered under two names is reported once
    param_names_by_storage = {
        StorageWeakRef(replacements[param_id].untyped_storage()): [names[0]]
        for param_id, names in param_names_by_param_id.items()
    }
    return originals, param_names_by_storage


def _get_buffer_names_by_storage(model: torch.nn.Module) -> dict[StorageWeakRef, list[str]]:
    buffer_names_by_storage = defaultdict(list)
    for name, buffer in model.named_buffers():
        buffer_names_by_storage[StorageWeakRef(buffer.untyped_storage())].append(name)
    return buffer_names_by_storage


class _SourceRecordingMode(TorchDispatchMode):
    """Carries through every op the HF names a tensor derives from, records them at each write into a param, and
    records every buffer written."""

    def __init__(
        self,
        param_names_by_storage: Mapping[StorageWeakRef, list[str]],
        buffer_names_by_storage: Mapping[StorageWeakRef, list[str]],
    ) -> None:
        super().__init__()
        self._param_names_by_storage = param_names_by_storage
        self._buffer_names_by_storage = buffer_names_by_storage
        # by identity, dropped when the tensor goes
        self._hf_names_by_tensor: WeakIdKeyDictionary = WeakIdKeyDictionary()
        self._hf_names_of_pending_scalar: frozenset[str] = frozenset()
        self.hf_names_by_param_name: dict[str, set[str]] = defaultdict(set)
        self.written_buffer_names: set[str] = set()

    def tag(self, tensor: torch.Tensor, hf_names: frozenset[str]) -> None:
        self._hf_names_by_tensor[tensor] = hf_names

    def _hf_names_of(self, value: object) -> frozenset[str]:
        return self._hf_names_by_tensor.get(value, frozenset()) if isinstance(value, torch.Tensor) else frozenset()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        hf_names = frozenset().union(*map(self._hf_names_of, tree_flatten((args, kwargs))[0]))
        if func is torch.ops.aten._local_scalar_dense.default:
            # the write that consumes the value takes its HF names
            self._hf_names_of_pending_scalar |= hf_names
            if not args[0].is_meta:
                return func(*args, **kwargs)
            # a param's value; where bytes land never depends on it
            dtype = args[0].dtype
            return False if dtype == torch.bool else (0.0 if dtype.is_floating_point else 0)
        written = [
            args[index] if index < len(args) else kwargs.get(argument.name)
            for index, argument in enumerate(func._schema.arguments)
            if argument.alias_info is not None and argument.alias_info.is_write
        ]
        if written and not hf_names:
            hf_names, self._hf_names_of_pending_scalar = self._hf_names_of_pending_scalar, frozenset()
        output = func(*args, **kwargs)
        for destination in written:
            if not isinstance(destination, torch.Tensor):
                continue
            storage = StorageWeakRef(destination.untyped_storage())
            param_names = self._param_names_by_storage.get(storage)
            if param_names is None:
                self.tag(destination, self._hf_names_of(destination) | hf_names)
                # most destinations are intermediates, in no table
                self.written_buffer_names.update(self._buffer_names_by_storage.get(storage, ()))
            else:
                # writes without HF names, such as zeroed padding, are not loaded data
                for param_name in param_names if hf_names else ():
                    self.hf_names_by_param_name[param_name] |= hf_names
        if hf_names:
            for value in tree_flatten(output)[0]:
                if isinstance(value, torch.Tensor) and not any(value is destination for destination in written):
                    self.tag(value, hf_names)
        return output
