import shlex
from pathlib import Path

import yaml
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).parents[3]
WORKFLOW_PATH = ROOT / ".github/workflows/_run-ci.yml"
DUMPS_ROOT_ENV = "MILES_TEST_DUMPS_ROOT"


class TestTheDumpsRootOfTheCiWorkflow:
    def test_every_job_names_an_absolute_dumps_root(self) -> None:
        """Without it every soak and comparison falls back to a path the CI host does not provide."""
        for name, job in _jobs().items():
            root = job["env"].get(DUMPS_ROOT_ENV)
            assert root is not None, f"job {name} sets no {DUMPS_ROOT_ENV}"
            assert Path(root).is_absolute()

    def test_the_dumps_root_sits_on_a_host_directory_mounted_at_the_same_path(self) -> None:
        """Dumps written inside the container's own filesystem vanish with it and cannot be collected."""
        for name, job in _jobs().items():
            root = Path(job["env"][DUMPS_ROOT_ENV])
            mounts = _identity_mounts(job["container"]["options"])
            assert any(root.is_relative_to(mount) for mount in mounts), f"job {name} writes {root} outside {mounts}"


def _jobs() -> dict[str, dict]:
    jobs = yaml.safe_load(WORKFLOW_PATH.read_text())["jobs"]
    assert jobs
    return jobs


def _identity_mounts(options: str) -> list[Path]:
    tokens = shlex.split(options)
    volumes = [tokens[index + 1] for index, token in enumerate(tokens) if token == "-v"]
    return [Path(host) for host, container in (volume.split(":")[:2] for volume in volumes) if host == container]
