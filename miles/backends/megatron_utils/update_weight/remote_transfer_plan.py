"""
Remote Transfer Plan - Abstract transfer planning for NCCL and RDMA weight updates.

This module provides a unified interface for determining transfer sources and planning
weight transfer tasks across different communication backends (NCCL, RDMA).
"""

import logging
from argparse import Namespace
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
from megatron.core import mpu

logger = logging.getLogger(__name__)


@dataclass
class TransferTaskP2PMeta:
    """Specifies a engine rollout rank to connect to."""

    engine_ind: int  # The index of the target rollout engine.
    engine_rank: int  # The shard within the target rollout engine.
    source_shard: int = 0  # The source pp shard index.


class RemoteTransferPlan:
    """
    Plans and manages remote weight transfers for both NCCL and RDMA backends,
    assuming static training and rollout placements.

    Key insight: the bucketed weight update all-gathers across TP/EP/ETP dimensions,
    so after all-gather every rank sharing the same PP rank holds a complete weight
    replica. The only remaining source-side parallelism axis is PP.

        gathered_dp_size = world_size / pp_size
        gathered_dp_rank = unique index [0, gathered_dp_size) per PP stage

    NCCL Plan: Broadcast from gathered_dp_rank=0 per PP stage to all rollout engines.
    RDMA P2P Plan: Round-robin assign gathered_dp_ranks to rollout (engine, rank) targets.
    """

    def __init__(
        self, args: Namespace, model: Sequence[torch.nn.Module], mode: Literal["nccl", "rdma"] = "nccl"
    ) -> None:
        self.mode = mode
        self._get_parallelism(args)

    def _get_parallelism(self, args: Namespace) -> None:
        """
        Collect parallelism information for source (trainer) and target (rollout engines).

        After the bucketed all-gather across TP/EP/ETP dimensions, every rank sharing
        the same PP rank holds a complete weight replica. So the effective source
        parallelism is simply: all ranks with the same PP rank.

            gathered_dp_size = world_size / pp_size
            gathered_dp_rank = unique index [0, gathered_dp_size) within that group
        """
        self._pp_rank = mpu.get_pipeline_model_parallel_rank()
        self._pp_size = mpu.get_pipeline_model_parallel_world_size()

        world_size = dist.get_world_size()
        self._size = world_size // self._pp_size

        # Rank within the gathered DP group.
        global_rank = dist.get_rank()
        my_pp_group = dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group())
        my_column_id = min(my_pp_group)

        all_column_ids = [None] * world_size
        dist.all_gather_object(all_column_ids, my_column_id)
        sorted_columns = sorted(set(all_column_ids))
        self._rank = sorted_columns.index(my_column_id)

        # Target (rollout engine) parallelism.
        self._rollout_tp_size = args.sglang_tp_size
        self._rollout_dp_size = args.sglang_dp_size
        self._rollout_ep_size = args.sglang_ep_size
        self._rollout_attn_tp_size = self._rollout_tp_size // self._rollout_dp_size
        self._rollout_moe_tp_size = self._rollout_tp_size // self._rollout_ep_size

        self._rollout_pp_size = args.sglang_pp_size
        if self._rollout_pp_size != 1:
            raise NotImplementedError("Rollout pipeline parallelism is not supported yet.")
        self._rollout_num_gpu_per_engine = args.rollout_num_gpus_per_engine
        self._rollout_engine_count = args.rollout_num_gpus // self._rollout_num_gpu_per_engine
        self._rollout_num_gpus = args.rollout_num_gpus

        logger.info(
            f"RemoteTransferPlan initialized: mode={self.mode}, "
            f"pp_rank={self._pp_rank}/{self._pp_size}, "
            f"gathered_dp_rank={self._rank}/{self._size} (global_rank={global_rank})"
        )
        logger.info(
            f"Rollout engine count: {self._rollout_engine_count}, "
            f"tp_size={self._rollout_tp_size}, ep_size={self._rollout_ep_size}, "
            f"dp_size={self._rollout_dp_size}"
        )

    def get_nccl_group(self) -> str:
        """Get the NCCL group name for weight transfer."""
        assert self.mode == "nccl", "NCCL group only applicable for NCCL mode."
        return f"miles-pp_{self._pp_rank}"

    def plan_p2p(self) -> list[TransferTaskP2PMeta]:
        """
        Plan P2P transfer tasks mapping source dp ranks to target (engine, rank) pairs.

        Heuristics:
        1. Round-robin assign source ranks to targets until all sources used at least once.
        2. For remaining targets, prioritize sources that already have the same engine_rank assigned.
        """
        all_targets = [
            (m_idx, k_idx)
            for m_idx in range(self._rollout_engine_count)
            for k_idx in range(self._rollout_num_gpu_per_engine)
        ]
        # Assignments: source_rank -> {engine_rank: [engine_indices]}
        assignments = defaultdict(lambda: defaultdict(list))

        # First round-robin assignment
        i = -1
        for source_rank, (idx, target) in zip(range(self._size), enumerate(all_targets), strict=False):
            i = idx
            m_idx, k_idx = target
            assignments[source_rank][k_idx].append(m_idx)

        def count_engine_index_assignments(k_idx: int) -> list[int]:
            return [len(assignments[source][k_idx]) for source in range(self._size)]

        # Remainder assignment by least-assigned source
        cur_source_index = 0
        if i < len(all_targets) - 1:
            for target in all_targets[i + 1 :]:
                m_idx, k_idx = target
                counted = count_engine_index_assignments(k_idx)
                if max(counted) > 0:
                    _, select_source = min((val, idx) for (idx, val) in enumerate(counted) if val > 0)
                else:
                    select_source = cur_source_index % self._size
                    cur_source_index += 1
                assignments[select_source][k_idx].append(m_idx)

        # Extract transfer tasks for current rank
        transfer_tasks = []
        for engine_rank, engine_indices in assignments[self._rank].items():
            for engine_ind in engine_indices:
                transfer_tasks.append(
                    TransferTaskP2PMeta(source_shard=self._pp_rank, engine_ind=engine_ind, engine_rank=engine_rank)
                )

        by_rank = defaultdict(list)
        for t in transfer_tasks:
            by_rank[t.engine_rank].append(t.engine_ind)
        dest_lines = [f"    engine_rank={r} -> engine(s) {sorted(by_rank[r])}" for r in sorted(by_rank)]
        logger.info(
            f"[TransferPlanner] Plan for gathered_dp_rank={self._rank} (pp={self._pp_rank}): "
            f"{self._size} source replicas, {len(transfer_tasks)} transfer(s) to "
            f"{self._rollout_engine_count} engine(s) x {self._rollout_num_gpu_per_engine} ranks/engine\n"
            + "\n".join(dest_lines)
        )
        return transfer_tasks

    def is_source(self) -> bool:
        """
        Determine if the current rank needs to initiate weight transfer.

        NCCL: only gathered_dp_rank=0 per PP stage broadcasts.
        RDMA: every gathered_dp_rank mapping to a rollout GPU is a source.
        """
        if self.mode == "nccl":
            return self._rank == 0
        return self._rank < self._rollout_num_gpus
