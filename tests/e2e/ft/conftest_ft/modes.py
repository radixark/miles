# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

from dataclasses import dataclass

import typer

MODEL_NAME: str = "Qwen3-30B-A3B-5layer"
MODEL_HF_REPO: str = f"fzyzcjy/{MODEL_NAME}"
MODEL_TYPE: str = "qwen3-30B-A3B-5layer"
DEBUG_ROLLOUT_DATA_HF_REPO: str = "fzyzcjy/miles-test-rollout-Qwen3-30B-A3B-5layer"

FULL_MODEL_NAME: str = "Qwen3-30B-A3B"
FULL_MODEL_HF_REPO: str = f"Qwen/{FULL_MODEL_NAME}"
FULL_MODEL_TYPE: str = "qwen3-30B-A3B"

# Small real dense model for with_failure's real_rollout mode (see README mode table).
DENSE_MODEL_NAME: str = "Qwen3-0.6B"
DENSE_MODEL_HF_REPO: str = f"Qwen/{DENSE_MODEL_NAME}"
DENSE_MODEL_TYPE: str = "qwen3-0.6B"


@dataclass(frozen=True)
class FTTestMode:
    model_name: str
    model_hf_repo: str
    megatron_model_type: str
    num_cells: int
    parallel_args: str
    train_num_nodes: int = 1
    train_gpus_per_node: int = 8
    rollout_num_engines: int = 0
    rollout_gpus_per_engine: int = 0
    num_steps: int = 10

    @property
    def has_real_rollout(self) -> bool:
        return self.rollout_num_engines > 0

    @property
    def total_rollout_gpus(self) -> int:
        return self.rollout_num_engines * self.rollout_gpus_per_engine

    @property
    def total_node_gpus(self) -> int:
        return self.train_gpus_per_node + self.total_rollout_gpus


MODES: dict[str, FTTestMode] = {
    # --- 1-node (8 GPUs) variants ---
    "dp2_cp2_tp2_ep2": FTTestMode(
        model_name=MODEL_NAME,
        model_hf_repo=MODEL_HF_REPO,
        megatron_model_type=MODEL_TYPE,
        num_cells=2,
        parallel_args=(
            "--tensor-model-parallel-size 2 "
            "--context-parallel-size 2 "
            "--expert-model-parallel-size 2 "
            "--sequence-parallel"
        ),
    ),
    "dp2_cp2_pp2": FTTestMode(
        model_name=MODEL_NAME,
        model_hf_repo=MODEL_HF_REPO,
        megatron_model_type=MODEL_TYPE,
        num_cells=2,
        parallel_args=(
            "--pipeline-model-parallel-size 2 "
            "--context-parallel-size 2 "
            "--decoder-first-pipeline-num-layers 3 "
            "--decoder-last-pipeline-num-layers 2"
        ),
    ),
    "dp4_cp2": FTTestMode(
        model_name=MODEL_NAME,
        model_hf_repo=MODEL_HF_REPO,
        megatron_model_type=MODEL_TYPE,
        num_cells=4,
        parallel_args="--context-parallel-size 2",
    ),
    "dp2_cp2_real_rollout": FTTestMode(
        model_name=MODEL_NAME,
        model_hf_repo=MODEL_HF_REPO,
        megatron_model_type=MODEL_TYPE,
        num_cells=2,
        train_gpus_per_node=4,
        rollout_num_engines=4,
        rollout_gpus_per_engine=1,
        parallel_args="--context-parallel-size 2",
    ),
    # Same topology as dp2_cp2_real_rollout but a small real dense model (see README).
    "dp2_cp2_real_rollout_dense": FTTestMode(
        model_name=DENSE_MODEL_NAME,
        model_hf_repo=DENSE_MODEL_HF_REPO,
        megatron_model_type=DENSE_MODEL_TYPE,
        num_cells=2,
        train_gpus_per_node=4,
        rollout_num_engines=4,
        rollout_gpus_per_engine=1,
        parallel_args="--context-parallel-size 2",
    ),
    # --- 6-node (48 GPUs) disaggregated: 4 train nodes + 2 rollout nodes ---
    "6node_dp4_cp2_tp2_pp2_ep2_etp2": FTTestMode(
        model_name=FULL_MODEL_NAME,
        model_hf_repo=FULL_MODEL_HF_REPO,
        megatron_model_type=FULL_MODEL_TYPE,
        num_cells=4,
        train_num_nodes=4,
        train_gpus_per_node=8,
        rollout_num_engines=2,
        rollout_gpus_per_engine=8,
        parallel_args=(
            "--tensor-model-parallel-size 2 "
            "--context-parallel-size 2 "
            "--pipeline-model-parallel-size 2 "
            "--expert-model-parallel-size 2 "
            "--expert-tensor-parallel-size 2 "
            "--sequence-parallel"
        ),
    ),
}


def resolve_mode(mode: str) -> FTTestMode:
    if mode not in MODES:
        raise typer.BadParameter(f"Unknown mode {mode!r}, valid: {list(MODES.keys())}")
    return MODES[mode]
