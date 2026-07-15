from miles.utils.async_utils import eager_create_task


async def train_actor_critic_models(args, actor_model, critic_model, rollout_id, rollout_data_ref):
    if args.use_critic:
        critic_task = await eager_create_task(critic_model.train(rollout_id, rollout_data_ref))
        try:
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_ref)
        finally:
            await critic_task
    else:
        await actor_model.train(rollout_id, rollout_data_ref)
