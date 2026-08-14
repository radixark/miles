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
    colocate: bool = False
    ft_components: tuple[str, ...] = ("train",)
    num_steps: int = 10

    def __post_init__(self) -> None:
        assert not self.colocate or self.total_rollout_gpus <= self.train_gpus_per_node, (
            f"Colocated mode oversubscribes its node: {self.total_rollout_gpus} rollout gpus "
            f"do not fit in {self.train_gpus_per_node} train gpus"
        )

    @property
    def has_real_rollout(self) -> bool:
        return self.rollout_num_engines > 0

    @property
    def total_rollout_gpus(self) -> int:
        return self.rollout_num_engines * self.rollout_gpus_per_engine

    @property
    def total_node_gpus(self) -> int:
        if self.colocate:
            return self.train_gpus_per_node
        return self.train_gpus_per_node + self.total_rollout_gpus


MODES: dict[str, FTTestMode] = {
    # --- 1-node (8 GPUs) variants ---
    "kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer": FTTestMode(
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
    "kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer": FTTestMode(
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
    "kill_train__dp4_cp2__fake_rollout__moe_5layer": FTTestMode(
        model_name=MODEL_NAME,
        model_hf_repo=MODEL_HF_REPO,
        megatron_model_type=MODEL_TYPE,
        num_cells=4,
        parallel_args="--context-parallel-size 2",
    ),
    "kill_train__dp2_cp2__moe_5layer": FTTestMode(
        model_name=MODEL_NAME,
        model_hf_repo=MODEL_HF_REPO,
        megatron_model_type=MODEL_TYPE,
        num_cells=2,
        train_gpus_per_node=4,
        rollout_num_engines=4,
        rollout_gpus_per_engine=1,
        parallel_args="--context-parallel-size 2",
    ),
    # Same topology as kill_train__dp2_cp2__moe_5layer but a small real dense model (see README).
    "kill_train__dp2_cp2": FTTestMode(
        model_name=DENSE_MODEL_NAME,
        model_hf_repo=DENSE_MODEL_HF_REPO,
        megatron_model_type=DENSE_MODEL_TYPE,
        num_cells=2,
        train_gpus_per_node=4,
        rollout_num_engines=4,
        rollout_gpus_per_engine=1,
        parallel_args="--context-parallel-size 2",
    ),
    # Same topology again, with ft on both kinds so one run crashes trainer cells and engines.
    "kill_train_rollout__dp2_cp2": FTTestMode(
        model_name=DENSE_MODEL_NAME,
        model_hf_repo=DENSE_MODEL_HF_REPO,
        megatron_model_type=DENSE_MODEL_TYPE,
        num_cells=2,
        train_gpus_per_node=4,
        rollout_num_engines=4,
        rollout_gpus_per_engine=1,
        ft_components=("train", "rollout"),
        parallel_args="--context-parallel-size 2",
    ),
    # --- 1-node (8 GPUs) colocated: engines share the trainer's gpus ---
    "kill_rollout__dp2_cp2__colocate": FTTestMode(
        model_name=DENSE_MODEL_NAME,
        model_hf_repo=DENSE_MODEL_HF_REPO,
        megatron_model_type=DENSE_MODEL_TYPE,
        num_cells=2,
        train_gpus_per_node=4,
        rollout_num_engines=4,
        rollout_gpus_per_engine=1,
        colocate=True,
        ft_components=("rollout",),
        parallel_args="--context-parallel-size 2",
    ),
    # --- 6-node (48 GPUs) disaggregated: 4 train nodes + 2 rollout nodes ---
    "kill_train__dp4_cp2_tp2_pp2_ep2_etp2__moe_full": FTTestMode(
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
