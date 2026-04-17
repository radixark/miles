import logging
import sys

import ray

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking


def train(args):
    configure_logger()
    logger = logging.getLogger(__name__)
    # allocate the GPUs
    logger.info("train: creating placement groups")
    pgs = create_placement_groups(args)
    logger.info("train: placement groups ready")
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    logger.info("train: creating rollout manager")
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    logger.info("train: rollout manager ready")

    # create the actor and critic models
    logger.info("train: creating training models")
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    logger.info("train: training models ready")

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    # actor_model.update_weights()

    # if args.check_weight_update_equal:
    #     ray.get(rollout_manager.check_weights.remote(action="compare"))

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train():
        if args.offload_train:
            actor_model.offload()
        else:
            actor_model.clear_memory()

    def save(rollout_id):
        actor_model.save_model(
            rollout_id,
            force_sync=rollout_id == args.num_rollout - 1,
        )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        logger.info(f"train: rollout {rollout_id} generate start")
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        #generating rollout data
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        logger.info(f"train: rollout {rollout_id} generate done")

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        logger.info(f"train: rollout {rollout_id} actor train start")
        ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
        logger.info(f"train: rollout {rollout_id} actor train done")

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        offload_train()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        # actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    # Ensure stdout is line-buffered so nohup logs show progress immediately.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    train(args)
