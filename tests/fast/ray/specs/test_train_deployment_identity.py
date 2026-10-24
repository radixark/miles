from tests.fast.ray.specs.test_train import _controller_context, _controller_providers, _make_args

from miles.ray.specs.train import specs_trainer_controller


class TestTrainerControllerDeploymentIdentity:
    def test_the_identity_records_the_declared_trainer_id(self) -> None:
        """A split trainer controller must identify which trainer entry it serves."""
        capability = _controller_providers()
        spec = specs_trainer_controller(_make_args(use_critic=True))[1]

        identity = spec.ctor_kwargs(_controller_context(capability))["deployment_identity"]

        assert identity.trainer_id == "critic"
