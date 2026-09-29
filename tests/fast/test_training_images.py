import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / "docker" / "build.py"
INSTALL_SCRIPT = REPO_ROOT / "docker" / "install-kube-tools.sh"
DEFAULT_DOCKERFILE = "docker/Dockerfile"
INSTALL_STEP = "install-kube-tools.sh"


def build_module():
    spec = importlib.util.spec_from_file_location("miles_docker_build", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)

    script_dir = str(BUILD_SCRIPT.parent)
    sys.path.insert(0, script_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(script_dir)

    return module


def training_dockerfiles() -> list[Path]:
    paths = {config.get("dockerfile", DEFAULT_DOCKERFILE) for config in build_module().VARIANTS.values()}
    return sorted(REPO_ROOT / path for path in paths)


class TestClusterToolsInEveryTrainingImage:
    @pytest.mark.parametrize("dockerfile", training_dockerfiles(), ids=lambda path: path.name)
    def test_the_image_installs_the_kubernetes_clients(self, dockerfile):
        """The workbench installs releases with them and the launcher drives the run with them."""
        assert INSTALL_STEP in dockerfile.read_text(), f"{dockerfile.name} builds an image without kubectl and helm"

    @pytest.mark.parametrize("dockerfile", training_dockerfiles(), ids=lambda path: path.name)
    def test_the_image_takes_the_shared_step_rather_than_its_own_copy(self, dockerfile):
        """Two hand-copied install blocks drift, and the one that drifts is the one nobody runs locally."""
        text = dockerfile.read_text()

        assert "dl.k8s.io" not in text
        assert "get.helm.sh" not in text

    def test_the_shared_step_pins_both_clients_and_checks_what_it_downloaded(self):
        """An unpinned or unverified client is a silent cluster-wide behaviour change on the next rebuild."""
        script = INSTALL_SCRIPT.read_text()

        assert re.search(r'^KUBECTL_VERSION="v\d+\.\d+\.\d+"$', script, re.MULTILINE)
        assert re.search(r'^HELM_VERSION="v\d+\.\d+\.\d+"$', script, re.MULTILINE)
        assert script.count("sha256sum -c -") == 2

    def test_every_variant_of_the_build_script_is_covered(self):
        """A new variant pointing at a third Dockerfile must not quietly opt out of this check."""
        assert len(training_dockerfiles()) == 2
