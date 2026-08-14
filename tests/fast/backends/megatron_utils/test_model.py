import logging

from tests.fast.backends.megatron_utils.conftest import TrainOneStepEnv


class TestTrainOneStepStructuredLog:
    def test_train_one_step_emits_the_train_tag_in_its_structured_event(
        self, train_one_step_env: TrainOneStepEnv, caplog
    ):
        """Log consumers key train-step events off the train tag, so the caller must emit that tag with its fields."""
        from miles.backends.megatron_utils.model import train_one_step

        with caplog.at_level(logging.INFO, logger="miles.backends.megatron_utils.model"):
            train_one_step(
                args=train_one_step_env.args,
                rollout_id=7,
                step_id=3,
                data_iterator=train_one_step_env.data_iterator,
                model=train_one_step_env.model,
                optimizer=None,
                opt_param_scheduler=None,
                num_microbatches=1,
                num_rollouts=1,
                witness_info=None,
                attempt=2,
            )

        assert "train op=train_step rollout=7 step=3 attempt=2 outcome=NORMAL valid_step=true" in caplog.messages
