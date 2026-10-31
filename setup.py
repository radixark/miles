from setuptools import find_packages, setup


def _fetch_requirements(path):
    with open(path) as fd:
        return [r.strip() for r in fd.readlines() if r.strip() and not r.startswith("#")]


# Setup configuration
setup(
    author="miles Team",
    name="miles",
    version="0.1.3",
    packages=find_packages(include=["miles*", "miles_plugins*"]),
    include_package_data=True,
    package_data={
        "miles.dashboard": ["static/*"],
        "miles.utils.chat_template_utils.templates": ["*.jinja"],
    },
    install_requires=_fetch_requirements("requirements.txt"),
    extras_require={
        "fsdp": [
            "torch>=2.0",
        ],
        # torchtitan 0.3.0 needs torch>=2.13 but does not declare torch
        "torchtitan": [
            "torch>=2.13",
            "torchtitan==0.3.0",
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
        # the openenv e2b sandbox backend, imported only when that backend is
        # selected. 2.12 is the first release that sends the template name in
        # the `name` field, which self-hosted AgentENV reads; older ones send
        # the deprecated `alias` and every Template.build fails with 400
        "e2b": [
            "e2b>=2.12",
        ],
        "modal": [
            "modal>=1.5.5",
        ],
    },
    # sglang needs 3.11; nvidia-resiliency-ext 0.6 has no 3.13 wheel
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)
