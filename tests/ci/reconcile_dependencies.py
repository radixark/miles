"""Install CI requirements while retaining the image's cuDNN runtime."""

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

LARGE_PACKAGE_BYTES = 100 * 1024 * 1024
RUNTIME_PACKAGES = {"torch", "triton", "apex", "sglang-kernel"}
RUNTIME_PREFIXES = ("nvidia-", "flash-attn", "flashinfer-", "transformer-engine")


def package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def protected_packages() -> set[str]:
    protected = set()
    for distribution in importlib.metadata.distributions():
        name = package_name(distribution.metadata["Name"])
        size = sum(file.size or 0 for file in distribution.files or [])
        if name in RUNTIME_PACKAGES or name.startswith(RUNTIME_PREFIXES) or size >= LARGE_PACKAGE_BYTES:
            protected.add(name)
    return protected


def uv_changes(output: str) -> set[str]:
    changes = re.findall(r"^ ([+-]) ([\w.-]+)(?:==| @ ).+$", output, re.MULTILINE)
    counts = re.findall(r"^Would (uninstall|install) (\d+) packages?$", output, re.MULTILINE)
    if not counts:
        if not changes and "Would make no changes" in output:
            return set()
        raise RuntimeError("Unrecognized uv install plan; refusing to install")
    expected = dict(counts)
    if len(expected) != len(counts) or any(
        sum(sign == symbol for sign, _ in changes) != int(expected.get(action, 0))
        for action, symbol in (("install", "+"), ("uninstall", "-"))
    ):
        raise RuntimeError("Incomplete uv install plan; refusing to install")
    return {package_name(name) for _, name in changes}


def check_install_plan(command: list[str], directory: Path, *, use_uv: bool) -> None:
    protected = protected_packages()
    if use_uv:
        result = subprocess.run([*command, "--dry-run", "--color", "never"], capture_output=True, text=True)
        print(result.stdout + result.stderr, end="", flush=True)
        result.check_returncode()
        changes = uv_changes(result.stdout + result.stderr)
    else:
        report = directory / "pip-plan.json"
        subprocess.run([*command, "--dry-run", "--report", str(report)], check=True)
        plan = json.loads(report.read_text())
        if plan["version"] != "1":
            raise RuntimeError("Unrecognized pip install plan; refusing to install")
        changes = {package_name(item["metadata"]["name"]) for item in plan["install"]}

    # Do not remove or bypass this gate: CI must never reinstall image runtimes or large packages.
    # Update these packages in the Docker image instead, even for same-version reinstalls.
    if blocked := sorted(protected & changes):
        raise RuntimeError(f"CI dependency gate blocks reinstalling: {', '.join(blocked)}. Rebuild the image instead.")


def reconcile(requirements: list[str]) -> None:
    pins = []
    for package in ("nvidia-cudnn-cu12", "nvidia-cudnn-cu13"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        pins.append(f"{package}=={version}")

    with tempfile.TemporaryDirectory(prefix="miles-ci-dependencies-") as directory:
        if pins:
            # TE needs the image's cuDNN even when torch declares an older exact pin.
            # An override replaces that pin; a constraint would only make it conflict.
            overrides = Path(directory) / "overrides.txt"
            overrides.write_text("\n".join(pins) + "\n")
            command = ["uv", "pip", "install", "--python", sys.executable, "--overrides", str(overrides)]
            print(f"Preserving image cuDNN: {', '.join(pins)}", flush=True)
        else:
            command = [sys.executable, "-m", "pip", "install"]

        for path in requirements:
            install = [*command, "--break-system-packages", "-r", path]
            check_install_plan(install, Path(directory), use_uv=bool(pins))
            subprocess.run(install, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", nargs="+")
    reconcile(parser.parse_args().requirements)
