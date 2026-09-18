import pytest
from tests.fast.charts.utils import requires_helm, run_helm_lint


@requires_helm
class TestChartLint:
    def test_lint_passes_on_the_defaults(self):
        """The chart installs with no values at all beyond the cluster's own."""
        result = run_helm_lint()
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize(
        "overrides",
        [
            [
                "--set-json",
                'infra.volumes=[{"name":"cluster-storage","persistentVolumeClaim":{"claimName":"shared"},'
                '"mounts":[{"mountPath":"/cluster-storage"}]}]',
            ],
            ["--set-json", "infra.volumes=[]"],
            ["--set", "rbac.leaderWorkerSets=false"],
            ["--set", "rbac.create=false", "--set", "serviceAccount.name=preexisting"],
        ],
    )
    def test_lint_passes_for_every_supported_combination(self, overrides):
        """Linting renders NOTES.txt too, so every branch of every template has to hold up."""
        result = run_helm_lint(*overrides)

        assert result.returncode == 0, result.stdout + result.stderr
