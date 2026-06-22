import logging
import random

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel
from megatron.core.config import set_experimental_flag
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.training.global_vars import _build_tokenizer, set_args

from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.utils.hf_config import register_hf_config_aliases
from miles.utils.indep_dp import IndepDPInfo

from .indep_dp import create_indep_dp_group
from .parallel import create_megatron_parallel_state

logger = logging.getLogger(__name__)


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducability."""
    # Ensure that different pipeline MP stages get different seeds.
    seed = seed_ + (100 * get_parallel_state().pp.rank)
    # Ensure different data parallel ranks get different seeds
    if data_parallel_random_init:
        seed = seed + (10 * get_parallel_state().intra_dp.rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed, te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)


def _initialize_distributed(args, get_embedding_ranks=None, get_position_embedding_ranks=None):
    """Initialize torch.distributed and core model parallel."""
    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    mpu.initialize_model_parallel(
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_comm_backend=args.pipeline_model_parallel_comm_backend,
        context_parallel_size=args.context_parallel_size,
        hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
        expert_model_parallel_size=args.expert_model_parallel_size,
        num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        order="tp-cp-ep-dp-pp" if not args.use_tp_pp_dp_mapping else "tp-cp-ep-pp-dp",
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        create_gloo_process_groups=args.enable_gloo_process_groups,
    )


def init(
    args,
    indep_dp_store_addr: str | None = None,
    indep_dp_info: IndepDPInfo | None = None,
):
    if indep_dp_info is None:
        indep_dp_info = IndepDPInfo.create_trivial()

    set_args(args)
    if args.enable_experimental:
        logger.info("Enable megatron experimental")
        set_experimental_flag(True)

    # Pytorch distributed.
    _initialize_distributed(args)

    indep_dp = create_indep_dp_group(
        store_addr=indep_dp_store_addr,
        indep_dp_info=indep_dp_info,
        megatron_rank=dist.get_rank(),
        megatron_world_size=dist.get_world_size(),
    )

    set_parallel_state(create_megatron_parallel_state(indep_dp=indep_dp))

    # sanity check
    if getattr(args, "indep_dp", False):
        assert args.data_parallel_size == 1

    # Random seeds for reproducibility.
    if args.rank == 0:
        logger.info(f"> setting random seeds to {args.seed} ...")
    _set_random_seed(
        args.seed,
        args.data_parallel_random_init,
        args.te_rng_tracker,
        args.inference_rng_tracker,
    )
    register_hf_config_aliases()
    _build_tokenizer(args)
    # We won't use this. initialize to pass some validation in megatron.
    init_num_microbatches_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        args.data_parallel_size,
        args.decrease_batch_size_if_needed,
    )

    if args.deterministic_mode:
        if args.rank == 0:
            logger.info("> running in deterministic mode")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)

    if args.debug_deterministic_collective:
        assert not args.overlap_grad_reduce, "deterministic collectives require synchronous grad sync"

    if args.tp_comm_overlap:
        from megatron.training.initialize import _initialize_tp_communicators

        _initialize_tp_communicators()

    if getattr(args, "custom_megatron_init_path", None):
        from miles.utils.misc import load_function

        custom_init = load_function(args.custom_megatron_init_path)
        custom_init(args)


# TODO shall we use a simpler method to determine which rank to init wandb?
def is_megatron_main_rank():
    return (
        get_parallel_state().intra_dp_cp.rank == 0
        and get_parallel_state().tp.rank == 0
        and get_parallel_state().pp.rank == get_parallel_state().pp.size - 1
    )
