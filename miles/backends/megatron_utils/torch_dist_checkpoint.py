"""Read a Megatron torch_dist checkpoint without Megatron.

The offline converters under tools/ run outside a training job. The pickles inside a checkpoint
(the DCP metadata and the common state) reference Megatron classes, so reading them stubs those
classes out instead of importing Megatron.
"""

import argparse
import io
import os
import pickle

import torch
import torch.distributed.checkpoint as dist_cp
from typing_extensions import override

# ShardedObject("common_state", None, (1,), (0,)).unique_key: the one object radixark/Megatron-LM
# stores the non-sharded training state under, args included.
_COMMON_STATE_KEY = "common_state/shard_0_1"
# Where Megatron-LM wrote that state before it moved inside the DCP checkpoint.
_LEGACY_COMMON_STATE_FILE = "common.pt"


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


class _StubbedPickle:
    """The pickle-module surface torch.load reaches for, backed by the stubbing unpickler. Passing
    it keeps the stubs explicit instead of patching pickle process-wide."""

    Unpickler = UnpicklerWrapper

    @staticmethod
    def load(file, **kwargs):
        return UnpicklerWrapper(file, **kwargs).load()


def make_storage_meta():
    storage_meta = getattr(dist_cp, "StorageMeta", None)
    if storage_meta is not None:
        return storage_meta()
    return dist_cp.metadata.StorageMeta()


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = make_storage_meta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class _StubbedObjectLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    """DefaultLoadPlanner unpickles objects with the plain pickle module; this one stubs Megatron."""

    def __init__(self) -> None:
        # Either flattener replaces state_dict with a copy, and load_bytes would fill the copy.
        super().__init__(flatten_state_dict=False, flatten_sharded_tensors=False)

    @override
    def load_bytes(self, read_item: dist_cp.planner.ReadItem, value: io.BytesIO) -> None:
        self.state_dict[read_item.dest_index.fqn] = torch.load(value, weights_only=False, pickle_module=_StubbedPickle)


def load_checkpoint_args(checkpoint_dir: str) -> argparse.Namespace:
    """The training args a torch_dist checkpoint was written with, from whichever layout it has."""
    legacy_path = os.path.join(checkpoint_dir, _LEGACY_COMMON_STATE_FILE)
    if os.path.exists(legacy_path):
        with open(legacy_path, "rb") as legacy_file:
            return torch.load(legacy_file, weights_only=False, pickle_module=_StubbedPickle)["args"]

    reader = WrappedStorageReader(checkpoint_dir)
    if _COMMON_STATE_KEY not in reader.read_metadata().state_dict_metadata:
        raise FileNotFoundError(
            f"{checkpoint_dir} holds neither {_LEGACY_COMMON_STATE_FILE} nor a {_COMMON_STATE_KEY!r} object, "
            "so it is not a Megatron torch_dist checkpoint"
        )
    state_dict = {_COMMON_STATE_KEY: io.BytesIO()}
    dist_cp.load(state_dict, storage_reader=reader, planner=_StubbedObjectLoadPlanner(), no_dist=True)
    # A ShardedObject is serialized as the one-element list of its data.
    return state_dict[_COMMON_STATE_KEY][0]["args"]
