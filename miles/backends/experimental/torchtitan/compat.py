"""torch-2.11 compat for torchtitan main (pinned @7e3f2eb). Import BEFORE any torchtitan module.

torchtitan main tracks torch nightly; on torch 2.11 exactly two shallow gaps exist
(empirically probed on H200, torch 2.11.0+cu130):
  1. torch.distributed.fsdp lacks the nightly-only ``DataParallelMeshDims`` symbol that
     torchtitan/distributed/fsdp.py, full_dtensor.py and flux/parallelize.py import.
     A placeholder dataclass unblocks the entire qwen3/qwen3_5 model+parallelize+adapter
     chain. It is never instantiated as long as dp_mesh_dims/edp_mesh_dims stay None
     (fsdp.py only forwards them to fully_shard when non-None — full_dtensor mode, unused).
  2. ``torchdata`` is an ordinary pip dependency (needed only for components.checkpoint's
     dataloader import chain).

Known genuinely-broken-on-2.11 surface (do NOT enable): EP>1 branch of
apply_fsdp_to_decoder (nightly 2-arg _get_mesh_info + ShardPlacementResult), full_dtensor
mode, varlen attn kwargs (use attn_backend="flex").
"""

import dataclasses

TORCHTITAN_PINNED_COMMIT = "7e3f2eb"


def apply_torchtitan_compat() -> None:
    import torch.distributed.fsdp as _fsdp

    if not hasattr(_fsdp, "DataParallelMeshDims"):

        @dataclasses.dataclass
        class DataParallelMeshDims:  # placeholder; never instantiated on 2.11 paths
            shard: object = None
            replicate: object = None

        _fsdp.DataParallelMeshDims = DataParallelMeshDims

    try:
        import torchdata  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "torchtitan backend requires torchdata (pip install 'torchdata>=0.8.0')"
        ) from e


apply_torchtitan_compat()
