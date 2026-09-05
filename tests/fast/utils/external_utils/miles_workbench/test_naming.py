from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.external_utils.miles_workbench.naming import run_release_name
from miles.utils.workers.types import DeployComponent


class TestRunReleaseName:
    def test_run_release_name_preserves_the_component_and_instance_id(self) -> None:
        """A release generated for an instance parses back to the same deployment identity."""
        release = run_release_name(
            run_id="demo", deploy_component=DeployComponent.INFERENCE, deploy_instance_id="east"
        )

        parsed = ReleaseName.parse(release)

        assert parsed is not None
        assert parsed.run_id == "demo"
        assert parsed.deploy_component is DeployComponent.INFERENCE
        assert parsed.deploy_instance_id == "east"
