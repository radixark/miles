import os
from pathlib import Path

import pytest

import miles.utils.external_utils.command_utils as U


@pytest.mark.integration
def test_megatron_packing_qwen3_0_6b() -> None:
    hf_path = os.environ.get("MILES_PACKING_HF_PATH")
    if not hf_path:
        pytest.skip("Set MILES_PACKING_HF_PATH to run this integration test.")

    if not Path(hf_path).exists():
        pytest.skip(f"Checkpoint path not found: {hf_path}")

    model_type = os.environ.get("MILES_PACKING_MODEL_TYPE", "qwen3-0.6B")
    num_gpus = int(os.environ.get("MILES_PACKING_NUM_GPUS", "1"))
    tp_size = int(os.environ.get("MILES_PACKING_TP_SIZE", str(num_gpus)))
    pad_multiplier = int(os.environ.get("MILES_PACKING_PAD_MULTIPLIER", "128"))
    num_samples = int(os.environ.get("MILES_PACKING_NUM_SAMPLES", "8"))
    backward = bool(int(os.environ.get("MILES_PACKING_BACKWARD", "0")))

    if tp_size != num_gpus:
        pytest.skip("This test expects TP size to match num_gpus (DP=1).")

    backward_flag = "--backward" if backward else ""
    cmd = (
        f'source "{U.repo_base_dir}/scripts/models/{model_type}.sh" && '
        "PYTHONPATH=/root/Megatron-LM "
        f"torchrun --nproc-per-node {num_gpus} tests/utils/megatron_packing_runner.py "
        "${MODEL_ARGS[@]} "
        f"--tensor-model-parallel-size {tp_size} "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--hf-checkpoint {hf_path} "
        f"--pad-multiplier {pad_multiplier} "
        f"--num-samples {num_samples} "
        f"{backward_flag}"
    )
    U.exec_command(cmd)
