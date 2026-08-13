import os
import platform
import re
import sys
from pathlib import Path

from setuptools import find_namespace_packages, find_packages, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Optional bundling of the patched sglang / Megatron-LM sources.
#
# `third_party/sglang` and `third_party/Megatron-LM` are git submodules. They
# are populated only when someone opts in with `git submodule update --init`
# (or clones with `--recurse-submodules`). A plain `git clone` -- which is what
# docker/Dockerfile does before `pip install -e .` -- leaves them empty.
#
# When they are absent we build exactly the package we always built: miles and
# miles_plugins only. Nothing is bundled and no `package_dir` entry is emitted,
# so an editable install cannot shadow the sglang/Megatron-LM that the image
# already installed elsewhere.
#
# When they are present (the wheel-publishing path) their Python sources are
# bundled into the distribution, so `pip install miles-rl` yields a working
# miles, sglang, megatron.* and miles_megatron_plugins in one shot.
# ---------------------------------------------------------------------------
_SGLANG_SRC = _HERE / "third_party" / "sglang" / "python"
_MEGATRON_SRC = _HERE / "third_party" / "Megatron-LM"
_BUNDLE_THIRD_PARTY = (_SGLANG_SRC / "sglang").is_dir() and (_MEGATRON_SRC / "megatron").is_dir()

# The bundled sources are pure Python (sglang's compiled kernels ship
# separately as the `sgl-kernel` PyPI package), so the published wheel must be
# tagged `py3-none-any` or it installs on only one interpreter/platform.
# Off by default to preserve the existing local/`pip install -e .` behaviour;
# the publish workflow opts in.
_PURE_WHEEL = os.environ.get("MILES_PURE_WHEEL") == "1"


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


def _read_long_description():
    """Return the README so PyPI renders it on the project page."""
    return _HERE.joinpath("README.md").read_text(encoding="utf-8")


def _project_dependencies(pyproject):
    """Return `[project].dependencies` from a pyproject.toml, or [] if absent."""
    if not pyproject.is_file():
        return []

    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            tomllib = None

    if tomllib is not None:
        with open(pyproject, "rb") as fd:
            return tomllib.load(fd).get("project", {}).get("dependencies", [])

    # No TOML parser available. The dependency array is a flat list of quoted
    # strings, so recover it directly rather than silently shipping an empty
    # extra that would look installable and then fail at import time.
    text = pyproject.read_text(encoding="utf-8")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.M | re.S)
    if block is None:
        return []
    return re.findall(r"['\"]([^'\"]+)['\"]", block.group(1))


def _third_party_extras():
    """Runtime requirements of the bundled sglang and Megatron-LM.

    Bundling their source rather than depending on their published packages
    means their dependency metadata does not come along, so it is reproduced
    here -- read from the pinned submodules, so it tracks whatever commit
    third_party/* points at instead of being a hand-maintained list that rots.

    These stay out of install_requires on purpose. sglang pins CUDA-13 wheels
    (flashinfer, sglang-kernel, nvidia-cutlass-dsl, flash-attn-4) that cannot
    resolve on a CPU-only or non-Linux machine, and putting them in the base
    requirements would make `pip install miles-rl` fail outright there rather
    than install the parts that do work.
    """
    if not _BUNDLE_THIRD_PARTY:
        return [], []

    sglang_deps = _project_dependencies(_SGLANG_SRC / "pyproject.toml")
    megatron_deps = _project_dependencies(_MEGATRON_SRC / "pyproject.toml")
    # megatron.training imports triton at module load, but Megatron-LM's
    # pyproject deliberately declares it as `triton; sys_platform == 'never'`
    # -- an unsatisfiable marker, because its own images ship triton with
    # torch. A pip install has no such image, so request it explicitly.
    megatron_deps = list(megatron_deps) + ["triton; sys_platform == 'linux'"]
    return sglang_deps, megatron_deps


def _discover_packages():
    """Return (packages, package_dir), bundling third_party/* when available."""
    packages = find_packages(include=["miles*", "miles_plugins*"])
    package_dir = {}

    if not _BUNDLE_THIRD_PARTY:
        return packages, package_dir

    # `sglang` is a regular package; `megatron`, `megatron.legacy` and several
    # sub-packages have no __init__.py, so namespace discovery is required.
    packages += find_namespace_packages(where=str(_SGLANG_SRC), include=["sglang", "sglang.*"])
    packages += find_namespace_packages(where=str(_MEGATRON_SRC), include=["megatron", "megatron.*"])
    # miles_megatron_plugins lives inside the Megatron-LM repo and is packaged
    # with megatron-core there; bundle it from the same submodule.
    packages += find_namespace_packages(
        where=str(_MEGATRON_SRC),
        include=["miles_megatron_plugins", "miles_megatron_plugins.*"],
    )
    package_dir = {
        "sglang": "third_party/sglang/python/sglang",
        "megatron": "third_party/Megatron-LM/megatron",
        "miles_megatron_plugins": "third_party/Megatron-LM/miles_megatron_plugins",
    }
    return packages, package_dir


# Custom wheel class to modify the wheel name
class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = _PURE_WHEEL

    def get_tag(self):
        if _PURE_WHEEL:
            # py3 (any Python 3), none (no ABI constraint), any (any platform).
            return "py3", "none", "any"

        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        abi_tag = f"{python_version}"
        platform_tag = _get_platform_tag()

        return python_version, abi_tag, platform_tag


_packages, _package_dir = _discover_packages()
_sglang_deps, _megatron_deps = _third_party_extras()

# Setup configuration
setup(
    author="Miles Team",
    author_email="miles@radixark.ai",
    # The PyPI distribution is `miles-rl`; plain `miles` is taken by an
    # unrelated project. Import names are unaffected (`import miles`).
    name="miles-rl",
    version="0.0.3",
    description="Enterprise-grade reinforcement learning for large-scale model post-training.",
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/radixark/miles",
    project_urls={
        "Documentation": "https://miles.radixark.com/docs",
        "Source": "https://github.com/radixark/miles",
        "Issues": "https://github.com/radixark/miles/issues",
    },
    license="Apache-2.0",
    license_files=("LICENSE",),
    keywords="reinforcement-learning rlhf rl moe sglang megatron post-training",
    packages=_packages,
    package_dir=_package_dir,
    include_package_data=True,
    package_data={"miles.dashboard": ["static/*"]},
    install_requires=_fetch_requirements("requirements.txt"),
    extras_require={
        "fsdp": [
            "torch>=2.0",
        ],
        "mlflow": [
            "mlflow>=2.0",
        ],
        # standalone offline serving; the training image already has these via
        # sglang, and polars is a base requirement (used by the collector)
        "dashboard": [
            "fastapi>=0.135",
            "uvicorn>=0.41",
            "prometheus_client>=0.24",
        ],
        # What the bundled third-party sources need in order to actually
        # import. Linux + CUDA 13 only; see _third_party_extras().
        "sglang": _sglang_deps,
        "megatron": _megatron_deps,
        "all": _sglang_deps + _megatron_deps,
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
