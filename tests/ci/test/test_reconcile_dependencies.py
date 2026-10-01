import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "reconcile_dependencies.py"


def wheel(directory, name, version, requires=()):
    normalized = name.replace("-", "_")
    info = f"{normalized}-{version}.dist-info"
    with zipfile.ZipFile(directory / f"{normalized}-{version}-py3-none-any.whl", "w") as archive:
        archive.writestr(f"{normalized}/__init__.py", f'__version__ = "{version}"\n')
        archive.writestr(
            f"{info}/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires),
        )
        archive.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{info}/RECORD", "")


@pytest.mark.parametrize("cudnn", ["nvidia-cudnn-cu12", "nvidia-cudnn-cu13", None])
def test_real_resolver_preserves_image_runtime_and_installs_dependencies(tmp_path, cudnn):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel(wheels, "ci-leaf", "1.0")
    wheel(wheels, "ci-client", "1.0", ["ci-torch==1.0", "ci-leaf==1.0"])
    wheel(wheels, "ci-torch", "1.0", [f"{cudnn}==9.20.0.48"] if cudnn else [])
    if cudnn:
        wheel(wheels, cudnn, "9.20.0.48")
        wheel(wheels, cudnn, "9.22.0.52")

    environment = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    python = str(environment / "bin" / "python")
    env = {
        **os.environ,
        "PIP_NO_INDEX": "1",
        "PIP_FIND_LINKS": str(wheels),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "UV_NO_INDEX": "1",
        "UV_FIND_LINKS": str(wheels),
    }
    seeded = ["ci-torch==1.0"] + ([f"{cudnn}==9.22.0.52"] if cudnn else [])
    subprocess.run([python, "-m", "pip", "install", "--no-deps", *seeded], env=env, check=True)
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("ci-client==1.0\n")

    if cudnn:
        baseline = subprocess.run(
            [python, "-m", "pip", "install", "--dry-run", "-r", str(requirements)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert f"{cudnn}-9.20.0.48" in baseline.stdout

    subprocess.run([python, str(SCRIPT), str(requirements)], env=env, check=True)
    installed = json.loads(subprocess.check_output([python, "-m", "pip", "list", "--format=json"], env=env, text=True))
    versions = {package["name"]: package["version"] for package in installed}
    assert versions["ci-client"] == versions["ci-leaf"] == "1.0"
    if cudnn:
        assert versions[cudnn] == "9.22.0.52"
    else:
        assert not any(name.startswith("nvidia-cudnn") for name in versions)

    requirements.write_text("ci-missing-dependency==1.0\n")
    failed = subprocess.run([python, str(SCRIPT), str(requirements)], env=env)
    assert failed.returncode != 0
