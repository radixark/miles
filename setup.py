from setuptools import find_namespace_packages, find_packages, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


def _fetch_requirements(path):
    with open(path) as fd:
        return [r.strip() for r in fd.readlines() if r.strip() and not r.startswith("#")]


def _read_long_description():
    """Return the README contents so PyPI renders the project page."""
    from pathlib import Path
    return Path(__file__).parent.joinpath("README.md").read_text(encoding="utf-8")


# Custom wheel class for the bundled miles-rl wheel.
#
# After Phase 6 bundles the patched sglang + Megatron-LM Python source
# from third_party/* into the miles wheel, the wheel contains *only*
# Python source — no compiled extensions are built during this packaging
# step (the GPU kernels in sglang are built at runtime via sgl-kernel,
# which is a separate PyPI package listed in extras_require["gpu"]).
#
# A pure-Python wheel must be tagged `py3-none-any` so it installs on any
# Python 3.x interpreter and any platform. Forcing a platform/ABI-specific
# tag (the previous behavior) made cp311 builds incompatible with the
# Python 3.12 production image.
class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        # No compiled extensions in this wheel.
        self.root_is_pure = True

    def get_tag(self):
        # py3 (any Python 3), none (no ABI constraint), any (any platform).
        return "py3", "none", "any"


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
    name="miles-rl",
    # Bump for each release. PyPI rejects re-uploads of the same version.
    version="0.0.2",
    author="Radixark Miles Team",
    author_email="miles@radixark.ai",
    description="Enterprise-grade reinforcement learning for large-scale model training.",
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/radixark/miles",
    project_urls={
        "Source": "https://github.com/radixark/miles",
        "Documentation": "https://www.radixark.com/miles/docs",
        "Issues": "https://github.com/radixark/miles/issues",
    },
    license="Apache-2.0",
    license_files=("LICENSE",),
    keywords="reinforcement-learning rlhf rl moe sglang megatron training",
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
