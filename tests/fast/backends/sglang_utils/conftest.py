from __future__ import annotations

import functools
import json
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any

_TINY_MODEL_CONFIG: dict[str, Any] = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 128,
    "initializer_range": 0.02,
    "intermediate_size": 256,
    "max_position_embeddings": 2048,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-05,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 1000,
}


@functools.lru_cache(maxsize=1)
def tiny_model_path() -> Path:
    model_path = Path(tempfile.mkdtemp(prefix="miles-tiny-model-"))
    (model_path / "config.json").write_text(json.dumps(_TINY_MODEL_CONFIG))
    return model_path


def make_engine_args(**overrides: Any) -> Namespace:
    """Args namespace covering every field ``_compute_server_args`` touches."""
    defaults: dict[str, Any] = dict(
        hf_checkpoint=str(tiny_model_path()),
        seed=42,
        offload_rollout=False,
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        fp16=False,
        lora_rank=0,
        lora_adapter_path=None,
        multi_lora=False,
        colocate=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)
