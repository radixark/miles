import sys
import platform

from setuptools import find_namespace_packages, find_packages, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


def _get_platform_tag():
    if platform.system() != "Linux":
        return platform.system().lower()

    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "manylinux1_x86_64"
    if machine in ("aarch64", "arm64"):
        return "manylinux2014_aarch64"
    return f"linux_{machine}"


def _fetch_requirements(path):
    with open(path) as fd:
        return [r.strip() for r in fd.readlines() if r.strip() and not r.startswith("#")]


# Custom wheel class to modify the wheel name
class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        abi_tag = f"{python_version}"
        platform_tag = _get_platform_tag()

        return python_version, abi_tag, platform_tag


# ----------------------------------------------------------------
# Bundle the patched sglang + Megatron-LM source from the git submodules
# at third_party/* directly into the miles wheel. After `pip install miles`,
# the user gets miles, sglang, megatron.*, and miles_megatron_plugins all
# at the top level of site-packages — no separate `radixark-sglang` /
# `radixark-megatron` packages, no private index for the patched forks.
#
# Trade-off: a user who runs `pip install upstream-sglang` after
# `pip install miles` will overwrite the patched sglang. This is
# documented in README; the assumption is that miles is the primary
# install in its target environment.
# ----------------------------------------------------------------
_miles_packages = find_packages(include=["miles*", "miles_plugins*"])
# `sglang` is a regular package (has __init__.py). find_packages would
# work, but find_namespace_packages is a superset and safer if the layout
# ever shifts.
_sglang_packages = find_namespace_packages(
    where="third_party/sglang/python", include=["sglang", "sglang.*"]
)
# `megatron`, `megatron.legacy`, and several sub-packages are namespace
# packages (no __init__.py). find_namespace_packages is required to pick
# them up.
_megatron_packages = find_namespace_packages(
    where="third_party/Megatron-LM", include=["megatron", "megatron.*"]
)
# miles_megatron_plugins lives inside the Megatron-LM repo (packaged with
# megatron-core upstream). Bundle it from the submodule so the miles-rl wheel
# still ships it at the top level of site-packages.
_megatron_plugins_packages = find_namespace_packages(
    where="third_party/Megatron-LM",
    include=["miles_megatron_plugins", "miles_megatron_plugins.*"],
)

# Setup configuration
setup(
    author="miles Team",
    name="miles-rl",
    version="0.2.1",
    packages=(
        _miles_packages
        + _sglang_packages
        + _megatron_packages
        + _megatron_plugins_packages
    ),
    package_dir={
        # Top-level miles/* and friends use the default (project root).
        # The bundled submodule sources need explicit mapping so setuptools
        # finds the files at their actual on-disk locations.
        "sglang": "third_party/sglang/python/sglang",
        "megatron": "third_party/Megatron-LM/megatron",
        "miles_megatron_plugins": "third_party/Megatron-LM/miles_megatron_plugins",
    },
    include_package_data=True,
    install_requires=_fetch_requirements("requirements.txt"),
    extras_require={
        "fsdp": [
            "torch>=2.0",
        ],
        "mlflow": [
            "mlflow>=2.0",
        ],
        # ----------------------------------------------------------------
        # Placeholders for the `pip install miles` roadmap (Phase 4).
        # These slots are intentionally empty at this stage; Phase 6 will
        # populate them with the dependencies that today are installed by
        # docker/Dockerfile rather than declared in setup.py.
        # ----------------------------------------------------------------
        # `cpu` — extras that only the lightweight (rollout/eval/CPU)
        # subset of miles needs, beyond install_requires. Likely empty
        # or near-empty.
        "cpu": [],
        # `gpu` — the heavy GPU stack: flash-attn, flash-attn-3,
        # flash-linear-attention, transformer-engine, apex, tilelang,
        # causal-conv1d, mamba-ssm, nvidia-modelopt, nvidia-cudnn-cu*,
        # torch_memory_saver, mbridge. Many require a CUDA toolchain at
        # install time; some are only available as pre-built wheels (see
        # yueming-yuan/miles-wheels) rather than from PyPI.
        "gpu": [],
        # `training` — the full training environment: everything in
        # `gpu` plus the patched sglang / Megatron-LM / Megatron-Bridge
        # that today are installed via git+url or COPY-from-submodule
        # in docker/Dockerfile. Will pull from a private Python index
        # once Phase 5 publishes the wheels.
        "training": [],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    cmdclass={"bdist_wheel": bdist_wheel},
)
