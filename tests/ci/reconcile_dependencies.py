"""Install CI requirements while retaining the image's cuDNN runtime."""

import argparse
import importlib.metadata
import subprocess
import sys
import tempfile
from pathlib import Path


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
            subprocess.run([*command, "--break-system-packages", "-r", path], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", nargs="+")
    reconcile(parser.parse_args().requirements)
