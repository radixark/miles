"""Example multi-LoRA training script.

Trains two LoRA adapters (gsm8k + dapo_math) simultaneously on the same base model.
Each adapter has its own dataset, reward function, and checkpoint directory.

Usage:
    ray start --head --num-gpus 8
    ray job submit -- python examples/multi_lora/train_multi_lora.py \
        --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate \
        --hf-checkpoint Qwen/Qwen2.5-0.5B-Instruct \
        --lora-rank 32 --target-modules all-linear \
        --multi-lora-dir examples/multi_lora/adapters \
        --multi-lora-n-adapters 4 \
        --rollout-batch-size 32 --global-batch-size 256 \
        --num-rollout 100
"""

import asyncio
from pathlib import Path

import ray

from miles.ray.multi_lora_controller import MultiLoRAController
from miles.ray.placement_group import (
    allocate_train_group,
    create_placement_groups,
    create_rollout_manager,
)
from miles.utils.arguments import parse_args
from miles.utils.async_utils import eager_create_task
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking


async def train(args):
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    # Create the multi-LoRA controller and register adapters
    controller = MultiLoRAController.remote(args.multi_lora_n_adapters, args.lora_rank)
    multi_lora_dir = Path(args.multi_lora_dir)
    for adapter_dir in sorted(multi_lora_dir.iterdir()):
        if (adapter_dir / "adapter.yaml").exists():
            ray.get(controller.register_run.remote(str(adapter_dir)))

    # Put controller on args for data source
    args.multi_lora_controller = controller
    args.data_source_path = "miles.rollout.multi_lora_data_source.MultiLoRADataSource"

    # Create rollout manager
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Allocate training group
    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
        role="actor",
        with_ref=args.kl_coef != 0 or args.use_kl_loss,
    )
    critic_model = None
    if args.use_critic:
        critic_model = allocate_train_group(
            args=args,
            num_nodes=args.critic_num_nodes,
            num_gpus_per_node=args.critic_num_gpus_per_node,
            pg=pgs["critic"],
            role="critic",
            with_ref=False,
        )
        critic_init_task = await eager_create_task(critic_model.init())

    # Set controller BEFORE init — workers need it to query adapter configs
    await actor_model.set_multi_lora_controller(controller)
    start_rollout_ids = await actor_model.init()

    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.use_critic:
        await critic_init_task
        await actor_model.connect(critic_model)

    await actor_model.set_rollout_manager(rollout_manager)
    if args.rollout_global_dataset:
        await rollout_manager.load.remote(args.start_rollout_id - 1)

    if args.offload_rollout:
        await rollout_manager.onload_weights.remote()

    await actor_model.update_weights()

    if args.offload_rollout:
        await rollout_manager.onload_kv.remote()

    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

    # Training loop (mirrors train.py including offload/onload)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            await rollout_manager.eval.remote(rollout_id)

        rollout_data_ref = await rollout_manager.generate.remote(rollout_id)

        if args.offload_rollout:
            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
            if "kv_cache" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            if "weight" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
            await rollout_manager.offload.remote(tags=offload_tags)

        await actor_model.train(rollout_id, rollout_data_ref)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            await actor_model.save_model(rollout_id)

        if args.offload_train:
            await actor_model.offload()
        else:
            await actor_model.clear_memory()

        if args.offload_rollout:
            await rollout_manager.onload_weights.remote()
        await actor_model.update_weights()
        if args.offload_rollout:
            await rollout_manager.onload_kv.remote()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            await rollout_manager.eval.remote(rollout_id)

    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(train(args))
