from miles.utils.metric_utils import namespace_metrics, strip_metrics_namespace


class TestNamespaceMetrics:
    def test_a_single_policy_run_keeps_the_names_it_had(self):
        """Every existing dashboard query is written against the unprefixed names."""
        log_dict, step_key = namespace_metrics(
            {"rollout/reward": 1.0}, trainer_model_id=None, step_name="rollout/step", step=7
        )

        assert log_dict == {"rollout/reward": 1.0, "rollout/step": 7}
        assert step_key == "rollout/step"

    def test_every_policy_gets_its_own_namespace_and_step_axis(self):
        """Two policies advance at their own pace, so one shared step axis would interleave their curves."""
        log_dict, step_key = namespace_metrics(
            {"rollout/reward": 1.0}, trainer_model_id="alpha", step_name="rollout/step", step=7
        )

        assert log_dict == {"alpha/rollout/reward": 1.0, "alpha/rollout/step": 7}
        assert step_key == "alpha/rollout/step"


class TestStripMetricsNamespace:
    def test_another_policys_namespace_is_left_unchanged(self) -> None:
        """Normalizing one policy must preserve every key owned by another policy."""
        metrics = {"alpha/train/loss": 1.0, "beta/train/loss": 2.0, "beta/train/step": 7}

        assert strip_metrics_namespace(metrics, trainer_model_id="alpha") == {
            "train/loss": 1.0,
            "beta/train/loss": 2.0,
            "beta/train/step": 7,
        }
