from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.platforms import tpu_shared_slice
from miles.utils import async_utils
from miles.utils.timer import timer


class UpdateWeightFromIci(WeightTransferProtocol):
    """Device-to-device weight sync over the ICI links of a TPU slice the trainer shares with one engine.

    Trainer rank i sends each bucket to engine TP rank i in one transfer, all ranks at once: the gathered dense
    tensors and rank i's own experts (the engine's EP shard of rank i), with their names, since those differ per
    rank. Each engine rank loads them through its own weight loader; no host copies.
    """

    # every trainer rank sends whole dense tensors, so it must hold every PP stage's
    required_placement = WeightUpdatePlacement(gather_pp=True)

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.group_name = "miles-ici"

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
        shared = tpu_shared_slice.spec()
        trainer_size = dist.get_world_size()
        if list(engine_gpu_counts or ()) != [trainer_size] or shared.size - shared.rank_offset != 2 * trainer_size:
            raise ValueError(
                "--update-weight-transfer-mode ici pairs trainer rank i with TP rank i of one engine that fills the "
                f"rest of the shared slice: trainer ranks {trainer_size}, engine devices {engine_gpu_counts}, "
                f"slice size {shared.size}"
            )
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, rollout_engines, engine_cell_ids)
        self.is_sender = True
        self._selector = selector
        self._engine_slice_rank = shared.rank_offset + trainer_size + dist.get_rank()
        if dist.get_rank() == 0:
            futures = [
                updater.submit_client_call(
                    "init_weights_update_group",
                    master_address=shared.store_host,
                    master_port=shared.store_port,
                    rank_offset=0,
                    world_size=shared.size,
                    group_name=self.group_name,
                    backend=tpu_shared_slice.ENGINE_BACKEND_NAME,
                )
                for updater in self.cell_updaters_of_cell_id.values()
            ]
            async_utils.wait_futures(futures)

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        names = [name for name, _ in bucket]
        futures = []
        if dist.get_rank() == 0:
            # posts the engine's receives; each engine rank reads its paired trainer rank's names from the store
            futures = [
                updater.submit_client_call(
                    "update_weights_from_distributed",
                    names=names,
                    dtypes=[tensor.dtype for _, tensor in bucket],
                    shapes=[tensor.shape for _, tensor in bucket],
                    selector=self._selector,
                    group_name=self.group_name,
                )
                for updater in self.cell_updaters_of_cell_id.values()
            ]
        packed = tpu_shared_slice.pack([tensor for _, tensor in bucket])
        with timer("ici_send"):
            tpu_shared_slice.send(packed, self._engine_slice_rank, names=names)
        with timer("ici_engine_load"):
            check_weight_sync_results(async_utils.wait_futures(futures), is_lora=False)
