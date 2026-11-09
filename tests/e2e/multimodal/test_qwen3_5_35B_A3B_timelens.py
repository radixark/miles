"""Qwen3.5-35B-A3B on the TimeLens video recipe, in token space.

Runs `examples/timelens_video/run_timelens_qwen3_5.py --model-name Qwen3.5-35B-A3B` as committed, cut to two
steps of 8 prompts. The data is the Charades shard of TimeLens-Bench alone (6.5 GB instead of the 210 GB of the
full recipe): 64 held-out items for eval and the next 256 for training, disjoint, both written with the
recipe's own `prepare_data.py` helpers. Functionality, not accuracy.
"""

from pathlib import Path

from examples.timelens_video.prepare_data import eval_rows, write
from examples.timelens_video.run_timelens_qwen3_5 import ScriptArgs, execute, prepare
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

register_cuda_ci(
    est_time=1200,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "multimodal"],
    hardware=["hopper"],
    num_gpus=8,
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")
register_ci_gate(metric_key="perf/actor_train_mfu")

# Its own data dir, so the CI parquets never shadow a full prepare_data.py run under /root/datasets/timelens.
DATA_DIR = "/root/datasets/timelens_ci"
EVAL_ITEMS = 64
TRAIN_ITEMS = 256


def _args() -> ScriptArgs:
    return ScriptArgs.from_env(
        model_name="Qwen3.5-35B-A3B",
        num_gpus_per_node=8,
        num_rollout=2,
        data_dir=DATA_DIR,
        extra_args="--rollout-batch-size 8 --global-batch-size 64 --ci-test ",
    )


def prepare_data(args: ScriptArgs):
    U = args.create_backend()
    root = Path(args.data_dir) / "timelens"
    bench = root / "TimeLens-Bench"
    U.exec_command_cpu(f"mkdir -p {bench}/videos")
    # The Charades annotations and shard directory out of the 74 GB repository; `--include` takes one pattern.
    U.exec_command_cpu(
        f"hf download --repo-type dataset TencentARC/TimeLens-Bench charades-timelens.json --local-dir {bench}"
    )
    U.exec_command_cpu(
        "hf download --repo-type dataset TencentARC/TimeLens-Bench "
        f'--include "video_shards/charades/*" --local-dir {bench}'
    )
    videos = bench / "videos"
    if not (videos / "charades").is_dir():
        # The shard carries charades/<vid>.mp4 entries, the layout eval_rows() reads. Extract once: the runner keeps
        # /root/datasets across runs, and the rename keeps a half-extracted directory from passing for a full one.
        U.exec_command_cpu(
            f"rm -rf {videos}.partial && mkdir -p {videos}.partial && "
            f"for t in {bench}/video_shards/charades/*.tar.gz; do tar xzf $t -C {videos}.partial || exit 1; done && "
            f"mv {videos}.partial/charades {videos}/charades && rm -rf {videos}.partial"
        )
    rows = eval_rows(root, "charades", None)
    write(rows[:EVAL_ITEMS], root / "timelens_bench_charades_600.parquet")
    write(rows[EVAL_ITEMS : EVAL_ITEMS + TRAIN_ITEMS], root / "timelens_grpo_train.parquet")


if __name__ == "__main__":
    args = _args()
    prepare(args)
    prepare_data(args)
    execute(args)
