from __future__ import annotations

from argparse import Namespace
from typing import Any


def make_engine_args(**overrides: Any) -> Namespace:
    """Args namespace covering every field ``_compute_server_args`` touches."""
    defaults: dict[str, Any] = dict(
        hf_checkpoint="/fake/model",
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
