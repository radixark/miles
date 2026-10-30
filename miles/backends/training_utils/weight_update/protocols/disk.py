import glob
import json
import logging
import os
import shutil
from argparse import Namespace
from collections.abc import Sequence

import safetensors.torch
import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.utils.data_replica import (
    OwnedBuckets,
    get_data_replica_rank_and_size,
)
from miles.backends.training_utils.weight_update.protocols.utils.owner import owner_is_gone, owner_tag
from miles.backends.training_utils.weight_update.protocols.utils.rank_errors import fail_on_every_rank
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.backends.training_utils.weight_update.session import (
    check_weight_sync_results,
    maybe_pause_engines,
    maybe_resume_engines,
)
from miles.utils import async_utils
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.hf_utils.checkpoint_files import copy_hf_metadata_files

logger = logging.getLogger(__name__)


class UpdateWeightFromDisk(WeightTransferProtocol):
    """
    Full-checkpoint weight sync over a shared filesystem. Each gathered HF bucket goes to the replica
    rank with the fewest bytes so far, which writes it as a safetensors shard of a fresh version dir, rank 0 adds
    the index and HF assets, and the engines reload it via /update_weights_from_disk. Needs no NCCL,
    CUDA IPC or /pull_weights, so it works with engines on another accelerator stack. A run writes its versions
    under `run-<owner_tag of rank 0>` in --update-weight-disk-dir, so runs sharing it never touch each other's.
    """

    # /update_weights_from_disk replaces every weight in one call, so no begin/end session
    use_weight_update_session = False

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._version_dir: str | None = None
        self._previous_version_dir: str | None = None
        self._weight_map: dict[str, str] = {}

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        engine_cell_ids: Sequence[str],
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, self.rollout_engines, engine_cell_ids)
        self.group_name = "miles-disk"
        rank0_tag = [owner_tag()]
        dist.broadcast_object_list(rank0_tag, src=0, group=get_gloo_group())
        self.run_dir = os.path.join(self.args.update_weight_disk_dir, f"run-{rank0_tag[0]}")
        replica_rank, replica_size = get_data_replica_rank_and_size(parallel_state, placement)
        self._owned_buckets = OwnedBuckets(replica_rank=replica_rank, replica_size=replica_size)
        self.is_sender = True

    def begin_sync(self, weight_version: int, iter_buckets) -> bool:
        self._version_dir = os.path.join(self.run_dir, f"weight_v{weight_version:06d}")
        self._weight_map = {}
        self._owned_buckets.reset()
        # the other ranks write only after the updater's barrier, so rank 0 alone prepares the directory
        with fail_on_every_rank("[disk] version directory"):
            if dist.get_rank() == 0:
                if self._previous_version_dir is None:
                    # a run ends by a kill, leaving its last version behind; on this host, delete those of gone runs
                    for run_dir in glob.glob(os.path.join(self.args.update_weight_disk_dir, "run-*")):
                        if owner_is_gone(os.path.basename(run_dir).removeprefix("run-")):
                            _remove_dir(run_dir)
                os.makedirs(self._version_dir, exist_ok=True)
        return True

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        self._owned_buckets.offer(bucket)

    def after_base_weights(self) -> None:
        # written only after the collective gathers finish, so the ranks' file writes run in parallel
        with fail_on_every_rank("[disk] shard write"):
            for bucket_index, bucket in self._owned_buckets.take():
                tensors = dict(bucket)
                shard = f"rank{dist.get_rank():03d}-shard{bucket_index:05d}.safetensors"
                safetensors.torch.save_file(tensors, os.path.join(self._version_dir, shard))
                self._weight_map.update({name: shard for name in tensors})

    def finalize(self, weight_version: int) -> None:
        weight_maps: list = [None] * dist.get_world_size()
        dist.all_gather_object(weight_maps, self._weight_map, group=get_gloo_group())
        with fail_on_every_rank("[disk] engine reload"):
            if dist.get_rank() == 0:
                self._write_index({name: shard for rank_map in weight_maps for name, shard in rank_map.items()})
                self._reload_engines(weight_version)
        self._drop_previous_version()

    def _write_index(self, weight_map: dict[str, str]) -> None:
        with open(os.path.join(self._version_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f)
        copy_hf_metadata_files(self.args.hf_checkpoint, self._version_dir)
        logger.info("[disk] wrote %d tensors to %s", len(weight_map), self._version_dir)

    def _reload_engines(self, weight_version: int) -> None:
        cell_updaters = list(self.cell_updaters_of_cell_id.values())
        maybe_pause_engines(self.args, cell_updaters)
        results = async_utils.wait_futures(
            [
                # bounded per engine, inside --update-weights-timeout, so a hung engine costs its cell, not the trainer
                updater.submit_client_call(
                    "update_weights_from_disk",
                    model_path=self._version_dir,
                    # the engine's own startup format otherwise applies, and a dummy-loaded engine cannot reload
                    load_format="safetensors",
                    weight_version=str(weight_version),
                )
                for updater in cell_updaters
            ]
        )
        check_weight_sync_results(results, is_lora=False)
        maybe_resume_engines(self.args, cell_updaters)

    def _drop_previous_version(self) -> None:
        if dist.get_rank() == 0 and self._previous_version_dir not in (None, self._version_dir):
            _remove_dir(self._previous_version_dir)
        self._previous_version_dir = self._version_dir


def _remove_dir(path: str) -> None:
    # a dir left behind holds a full checkpoint on the shared filesystem, so say so instead of failing the sync
    try:
        shutil.rmtree(path)
    except OSError as error:
        logger.warning("[disk] could not delete %s: %s", path, error)
