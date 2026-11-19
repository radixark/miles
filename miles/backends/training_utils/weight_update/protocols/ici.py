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
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.timer import timer


class UpdateWeightFromIci(WeightTransferProtocol):
    """Device-to-device weight sync over the ICI links of a TPU slice shared with K >= 1 uniform rollout engines.

    For K uniform engines of ``tp_per_engine`` devices each (``trainer_size = K * tp_per_engine``,
    ``slice_size = 2 * trainer_size``), trainer rank ``j`` pairs bijectively with ``(tp_rank, engine_idx) = divmod(j, K)``
    at slice rank ``trainer_size + engine_idx * tp_per_engine + tp_rank``. When ``K == 1``, ``divmod(j, 1) == (j, 0)``,
    reducing to 1:1 pairing between trainer rank ``j`` and engine TP rank ``j``. On multi-host slices where each
    trainer host owns a contiguous span of ranks, pairing ``tp_rank = j // K`` keeps all trainer ranks serving the
    same EP shard co-located within a contiguous rank subgroup.
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
        counts = list(engine_gpu_counts or ())
        if (
            shared is None
            or shared.rank_offset != 0
            or not counts
            or len(set(counts)) != 1
            or sum(counts) != trainer_size
            or shared.size != 2 * trainer_size
        ):
            raise ValueError(
                "--update-weight-transfer-mode ici requires uniform rollout engines summing to trainer_size in a "
                f"shared slice of size 2 * trainer_size: slice rank offset "
                f"{None if shared is None else shared.rank_offset} (must be 0), trainer ranks {trainer_size}, "
                f"engine devices {engine_gpu_counts}, slice size {None if shared is None else shared.size}"
            )
        tp_per_engine = int(counts[0])
        num_engines = len(counts)
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, rollout_engines, engine_cell_ids)
        self.is_sender = True
        self._selector = selector
        rank = dist.get_rank()
        tp_rank, engine_idx = divmod(rank, num_engines)
        self._engine_slice_rank = trainer_size + engine_idx * tp_per_engine + tp_rank
        tpu_shared_slice.reset_tags()
        if rank == 0:
            futures = [
                updater.submit_client_call(
                    "init_weights_update_group",
                    master_address=shared.store_host,
                    master_port=shared.store_port,
                    rank_offset=idx * tp_per_engine,
                    world_size=shared.size,
                    group_name=self.group_name,
                    backend=tpu_shared_slice.receiver_contract().BACKEND_NAME,
                )
                for idx, updater in enumerate(self.cell_updaters_of_cell_id.values())
            ]
            async_utils.wait_futures(futures)

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        names = [name for name, _ in bucket]
        # pack before asking the engine to enter its receives, so a refused dtype cannot strand it there
        packed = tpu_shared_slice.pack([tensor for _, tensor in bucket])
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
        with timer("ici_send"):
            tpu_shared_slice.send(packed, self._engine_slice_rank, names=names)
        with timer("ici_engine_load"):
            engine_error = None
            if futures:
                try:
                    check_weight_sync_results(async_utils.wait_futures(futures), is_lora=False)
                except Exception as error:  # re-raised below, once every rank knows
                    engine_error = error
            # only rank 0 hears from the engine; every rank must stop here, not block in the next bucket's send
            loaded = torch.tensor([engine_error is None], dtype=torch.int32)
            dist.all_reduce(loaded, op=dist.ReduceOp.MIN, group=get_gloo_group())
        if engine_error is not None:
            raise engine_error
        if not loaded.item():
            raise RuntimeError("the engine failed to load an ICI weight bucket (see trainer rank 0)")
