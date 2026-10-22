"""Run Qwen3-VL GEO3K RL validation on four temporary Modal H100s.

From the miles checkout, with Modal credentials configured:
    modal run examples/multi_lora/modal_geo3k.py --output-dir ./geo3k-results

The checkpoint is cached in a Modal Volume. GPUs stop when validation finishes.
"""

import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import modal

app = modal.App("miles-tinker-multimodal-validation")
model_cache = modal.Volume.from_name("miles-tinker-multimodal-models", create_if_missing=True)
image = (
    modal.Image.from_registry("radixark/miles@sha256:360616b7678698a8c6b358b57e80c18b8bd3758da3d4753adb20f95432e8a511")
    .entrypoint([])
    .run_commands(
        "git clone --filter=blob:none https://github.com/radixark/Megatron-LM.git /opt/miles-megatron"
        " && git -C /opt/miles-megatron checkout a84b105473eca9c9c6e49dbf1c8aa460c5cf796e"
        " && pip install --no-deps --no-build-isolation -e /opt/miles-megatron",
        "pip install --no-deps --no-build-isolation git+https://github.com/radixark/Megatron-Bridge.git@8cd3466d14d2337c8492827b3712482c2b3e4866",
    )
    .pip_install(
        "tinker==0.26.2",
        "fastapi",
        "uvicorn",
        "transformers==5.12.1",
        "opentelemetry-api==1.44.0",
        "opentelemetry-sdk==1.44.0",
        "opentelemetry-exporter-otlp==1.44.0",
    )
    .pip_install("datasets==5.0.1", "pylatexenc==2.11")
    .env({"PYTHONPATH": "/workspace/miles:/opt/miles-megatron", "CUDA_DEVICE_MAX_CONNECTIONS": "1"})
)
if modal.is_local():
    image = image.add_local_dir(
        Path(__file__).resolve().parents[2], "/workspace/miles", ignore=[".git", "__pycache__", ".pytest_cache"]
    )


@app.function(image=image, volumes={"/models": model_cache}, timeout=7200, gpu="H100:4", cpu=32, memory=196608)
def validate():
    from huggingface_hub import snapshot_download

    os.chdir("/workspace/miles")
    model_path = "/models/Qwen3-VL-30B-A3B-Instruct"
    snapshot_download(
        "Qwen/Qwen3-VL-30B-A3B-Instruct", revision="9c4b90e1e4ba969fd3b5378b57d966d725f1b86c", local_dir=model_path
    )
    model_cache.commit()
    output_dir = Path(f"/models/validation/geo3k-{uuid.uuid4().hex[:8]}")
    output_dir.mkdir(parents=True)
    print(f"Validation artifacts: {output_dir}", flush=True)
    env = dict(
        os.environ,
        MILES_MULTIMODAL_CHECKPOINT=model_path,
        MILES_MEGATRON_PATH="/opt/miles-megatron",
        MILES_GEO3K_OUTPUT_DIR=str(output_dir),
        HF_DATASETS_CACHE="/models/datasets",
    )
    try:
        subprocess.run(
            ["python", "-m", "examples.multi_lora.validate_geo3k"],
            env=env,
            check=True,
        )
    finally:
        model_cache.commit()
    return {path.name: path.read_text() for path in output_dir.iterdir()}


@app.local_entrypoint()
def main(output_dir: str = ""):
    artifacts = validate.remote()
    if artifacts:
        destination = Path(output_dir or tempfile.mkdtemp(prefix="miles-multimodal-results-"))
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in artifacts.items():
            (destination / name).write_text(content)
        print(f"Saved validation results to {destination}")
