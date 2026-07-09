import asyncio

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.async_utils import eager_create_task
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import finish_tracking, init_tracking


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
async def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    await actor_model.update_weights()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:
        await rollout_manager.eval.remote(0)

    # async train loop.
    # Eval/save are keyed on weight_update_count (not rollout_id) so they track
    # actual model changes — critical for fully async where rollout_id doesn't
    # correspond 1:1 with weight versions.
    weight_update_count = 0
    total_weight_updates = (args.num_rollout - args.start_rollout_id) // args.update_weights_interval
    num_weight_updates_per_epoch = (
        num_rollout_per_epoch // args.update_weights_interval if num_rollout_per_epoch is not None else None
    )
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = await rollout_data_next_future

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            critic_task = await eager_create_task(critic_model.train(rollout_id, rollout_data_curr_ref))
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_curr_ref)
            await critic_task
        else:
            await actor_model.train(rollout_id, rollout_data_curr_ref)

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = (await x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            await actor_model.update_weights()
            weight_update_count += 1

            is_last = rollout_id == args.num_rollout - 1
            if should_run_periodic_action(
                weight_update_count, args.save_interval, num_weight_updates_per_epoch, total_weight_updates
            ):
                await actor_model.save_model(
                    weight_update_count,
                    force_sync=is_last,
                )
                if args.use_critic:
                    await critic_model.save_model(
                        weight_update_count,
                        force_sync=is_last,
                    )
                if args.rollout_global_dataset:
                    await rollout_manager.save.remote(weight_update_count)

            if should_run_periodic_action(weight_update_count, args.eval_interval, num_weight_updates_per_epoch):
                await rollout_manager.eval.remote(weight_update_count)

    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()
