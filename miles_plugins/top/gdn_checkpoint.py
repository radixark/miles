"""HF loading for TOP's explicit Qwen3.5 and Qwen3.8-Flash-Next hybrid parameter ownership."""

from collections import OrderedDict
from pathlib import Path

import torch
from megatron.core.utils import unwrap_model
from safetensors import safe_open

from mbridge.core.safetensor_io import SafeTensorIO
from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.mbridge.qwen3_5 import Qwen3_5Bridge


class _PackedExpertCheckpointIO(SafeTensorIO):
    """Reuse the gate/up and down tensors while loading their individual experts."""

    def __init__(self, hf_dir):
        super().__init__(hf_dir)
        self._recent_weights = OrderedDict()
        self._open_shards = OrderedDict()

    def load_some_hf_weight(self, names):
        if not self.index:
            return super().load_some_hf_weight(names)
        names, mapping = self._mapping_weight_names_old2new(names)
        weights = {}
        for name in names:
            filename = self.index[name]
            if filename in self._open_shards:
                shard = self._open_shards.pop(filename)
            else:
                if len(self._open_shards) == 2:
                    _, old_shard = self._open_shards.popitem(last=False)
                    old_shard.__exit__(None, None, None)
                shard = safe_open(str(Path(self.hf_dir) / filename), framework="pt", device="cpu")
                shard.__enter__()
            self._open_shards[filename] = shard
            weights[mapping[name]] = shard.get_tensor(name)
        return weights

    def close(self):
        self._recent_weights.clear()
        for shard in self._open_shards.values():
            shard.__exit__(None, None, None)
        self._open_shards.clear()

    def load_one_hf_weight(self, name):
        if name in self._recent_weights:
            weight = self._recent_weights.pop(name)
        else:
            if len(self._recent_weights) == 2:
                self._recent_weights.popitem(last=False)
            weight = super().load_one_hf_weight(name)
        self._recent_weights[name] = weight
        return weight


class _GDNCheckpointBridge(Qwen3_5Bridge):
    # TopQwenGDN owns flat HF Q/K/V rows. The native head-sharded module
    # uses group-major rows, but that conversion must not run for TOP.
    _GDN_GROUP_MAJOR = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_io = None

    def _get_safetensor_io(self, weights_path):
        self.checkpoint_io = _PackedExpertCheckpointIO(self._get_actual_hf_path(weights_path))
        return self.checkpoint_io


def _qwen38_checkpoint_bridge_cls():
    from miles_plugins.mbridge.qwen3_8_next import Qwen38NextBridge

    class _Qwen38CheckpointBridge(Qwen38NextBridge):
        # Qwen3.8-Next already keeps HF's flat GDN rows; TOP keeps the packed-expert IO.
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.checkpoint_io = None

        def _get_safetensor_io(self, weights_path):
            self.checkpoint_io = _PackedExpertCheckpointIO(self._get_actual_hf_path(weights_path))
            return self.checkpoint_io

    return _Qwen38CheckpointBridge


def load_gdn_hf_weights(models, checkpoint):
    models = unwrap_model(models)
    for model in models:
        config = model.config
        # CP ranks hold full weights; only TP would shard them.
        if config.tensor_model_parallel_size != 1:
            raise NotImplementedError("GDN TOP checkpoint loading requires TP1")
    hf_config = load_hf_config(checkpoint)
    bridge_cls = _qwen38_checkpoint_bridge_cls() if hf_config.model_type == "qwen4_exp" else _GDNCheckpointBridge
    bridge = bridge_cls(hf_config, dtype=torch.bfloat16)
    try:
        with torch.no_grad():
            bridge.load_weights(models, checkpoint, memory_efficient=True)
    finally:
        if bridge.checkpoint_io is not None:
            bridge.checkpoint_io.close()
