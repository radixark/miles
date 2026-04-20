"""E2E test: verify PD disaggregation on/off produce identical routing expert
results across MoE models, after installing the ``pd-r3`` branch of
sgl-router-for-miles (PR #4).

Design
~~~~~~
For each model, run the same rollout workload twice under
``--debug-rollout-only --sglang-enable-deterministic-inference
--use-rollout-routing-replay``:

1. ``variant=pd_off``: ``--rollout-num-gpus-per-engine 1`` (single engine,
   no PD disaggregation).
2. ``variant=pd_on``: ``--prefill-num-servers 1`` with two rollout GPUs —
   one prefill server + one decode server. Requires the Rust gateway to
   include PR #4 so that ``return_routed_experts`` is propagated across
   the prefill/decode boundary.

Each run writes a JSONL of per-sample ``(tokens, rollout_log_probs,
rollout_routed_experts)`` via ``utils.router_equivalence_generate``; we
then diff the two dumps byte-for-byte.

Backend / checkpoint
~~~~~~~~~~~~~~~~~~~~
Megatron backend (same as the sibling ``tests/e2e/megatron/*_r3.py``
tests) so that ``scripts/models/{type}.sh`` populates ``args.num_layers``
and ``args.moe_router_topk`` — which the rollout-side
``routed_experts`` reshape needs.  We do *not* set ``--use-kl-loss`` or
``--kl-coef`` > 0, which is what gates the ``--ref-load`` existence
check in ``miles/utils/arguments.py``, and ``--debug-rollout-only``
makes ``_compute_megatron_num_gpus`` return ``0`` so no megatron actor
is spawned and the checkpoint is never loaded.  This lets us get away
with a single/dual H200 and no ``convert_hf_to_torch_dist`` step.

PD transport
~~~~~~~~~~~~
We stay on the default ``mooncake`` backend but pin
``--sglang-disaggregation-ib-device mlx5_0`` explicitly.  Without the
pin, mooncake's auto-discovery picks up ``mlx5_bond_0`` — a LAG device
whose UMR QP registration fails with
``ibv_modify_qp(UMR QP) failed … No such device``.  The mooncake path is
also what production uses (see ``scripts/run_glm5_744b_a40b.py``).

Controls
~~~~~~~~
- ``PD_EQ_MODEL_FAMILY``: ``qwen3_30b_a3b`` (default) | ``glm47_flash``.
- PD variant uses 2 H200s (1 prefill + 1 decode); PD-off uses 1 H200.
  Both variants keep TP=1 so deterministic inference should match.
"""

import base64
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import miles.utils.external_utils.command_utils as U

MODEL_FAMILY = os.environ.get("PD_EQ_MODEL_FAMILY", "qwen3_30b_a3b")
DUMP_ROOT = Path(os.environ.get("PD_EQ_DUMP_ROOT", "/tmp/pd-eq"))
PROMPT_DATA_PATH = "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl"
NUM_PROMPTS = int(os.environ.get("PD_EQ_NUM_PROMPTS", "10"))
MAX_RESPONSE_LEN = int(os.environ.get("PD_EQ_MAX_RESPONSE_LEN", "256"))

# Repo root (tests/e2e/sglang/test_*.py → parents[3]).  Used to prepend the
# miles repo onto the Ray actor PYTHONPATH so the custom generate function is
# importable regardless of where the worktree lives.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    hf_repo: str
    local_dir: str
    megatron_model_type: str
    reasoning_parser: str | None = None


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "qwen3_30b_a3b": ModelConfig(
        model_name="Qwen3-30B-A3B",
        hf_repo="Qwen/Qwen3-30B-A3B",
        local_dir="/root/models/Qwen3-30B-A3B",
        megatron_model_type="qwen3-30B-A3B",
        reasoning_parser=None,
    ),
    "glm47_flash": ModelConfig(
        model_name="GLM-4.7-Flash",
        hf_repo="zai-org/GLM-4.7-Flash",
        local_dir="/root/models/GLM-4.7-Flash",
        megatron_model_type="glm4.7-flash",
        reasoning_parser="glm45",
    ),
}


def _get_config() -> ModelConfig:
    if MODEL_FAMILY not in MODEL_REGISTRY:
        raise ValueError(f"Unknown PD_EQ_MODEL_FAMILY={MODEL_FAMILY!r}; choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[MODEL_FAMILY]


def prepare() -> None:
    cfg = _get_config()
    U.exec_command("mkdir -p /root/models /root/datasets")
    if not Path(cfg.local_dir).exists():
        U.exec_command(f"hf download {cfg.hf_repo} --local-dir {cfg.local_dir}")
    if not Path(PROMPT_DATA_PATH).exists():
        U.hf_download_dataset("zhuzilin/dapo-math-17k")


def _variant_dir(variant: str) -> Path:
    return DUMP_ROOT / MODEL_FAMILY / variant


def _variant_dump_path(variant: str) -> Path:
    return _variant_dir(variant) / "dump.jsonl"


def _build_train_args(cfg: ModelConfig, variant: str) -> tuple[str, int]:
    ckpt_args = f"--hf-checkpoint {cfg.local_dir} "

    rollout_args = (
        f"--prompt-data {PROMPT_DATA_PATH} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        f"--rollout-batch-size {NUM_PROMPTS} "
        "--n-samples-per-prompt 1 "
        f"--rollout-max-response-len {MAX_RESPONSE_LEN} "
        "--rollout-temperature 0.0 "
        f"--global-batch-size {NUM_PROMPTS} "
        "--rollout-seed 42 "
    )

    generate_args = "--custom-generate-function-path " "tests.e2e.sglang.utils.router_equivalence_generate.generate "

    router_args = "--use-rollout-routing-replay "

    # Megatron parallelism args — 1 GPU per engine, no parallelism.
    # Not consumed by anything under --debug-rollout-only beyond arg parsing.
    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
    )

    if variant == "pd_on":
        # Pin the mooncake HCA to the PIX-neighbour of GPU 0 (prefill). Without
        # the pin, mooncake's auto-discovery picks ``mlx5_bond_0`` — a LAG
        # device whose UMR QP registration fails. Decode (GPU 1, PIX-neighbour
        # ``mlx5_1``) shares the same HCA, which crosses one PXB but works
        # functionally.
        sglang_args = (
            "--rollout-num-gpus-per-engine 1 "
            "--prefill-num-servers 1 "
            "--sglang-disaggregation-ib-device mlx5_0 "
            "--sglang-enable-deterministic-inference "
            "--sglang-mem-fraction-static 0.85 "
        )
        num_gpus = 2
    elif variant == "pd_off":
        sglang_args = (
            "--rollout-num-gpus-per-engine 1 "
            "--sglang-enable-deterministic-inference "
            "--sglang-mem-fraction-static 0.85 "
        )
        num_gpus = 1
    else:
        raise ValueError(f"unknown variant {variant!r}")

    if cfg.reasoning_parser:
        sglang_args += f"--sglang-reasoning-parser {cfg.reasoning_parser} "

    infra_args = (
        "--debug-rollout-only "
        "--ci-test "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {num_gpus} "
        "--colocate "
    )

    return ckpt_args + rollout_args + generate_args + router_args + perf_args + sglang_args + infra_args, num_gpus


def _run_variant(cfg: ModelConfig, variant: str) -> None:
    dump_dir = _variant_dir(variant)
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = _variant_dump_path(variant)

    train_args, num_gpus = _build_train_args(cfg, variant)
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus,
        megatron_model_type=cfg.megatron_model_type,
        extra_env_vars={
            # Ray actor default PYTHONPATH is Megatron-only; prepend the miles
            # repo so the custom generate function under ``tests.e2e.sglang.*``
            # is importable inside ``RolloutManager``.
            "PYTHONPATH": f"{_REPO_ROOT}:/root/Megatron-LM",
            "MILES_ROUTER_EQ_DUMP_PATH": str(dump_path),
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        },
    )


def _load_dump(path: Path) -> list[dict]:
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda r: r["index"])
    return records


def _assert_records_equal(left: list[dict], right: list[dict]) -> None:
    assert len(left) == len(right), f"dump length differs: {len(left)} vs {len(right)}"

    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        assert a["index"] == b["index"], f"record {i}: index {a['index']} vs {b['index']}"

        for field in ("status", "response_length", "tokens"):
            assert (
                a[field] == b[field]
            ), f"index={a['index']} field={field} mismatch:\n  pd_off: {a[field]}\n  pd_on:  {b[field]}"

        la = a["rollout_log_probs"] or []
        lb = b["rollout_log_probs"] or []
        assert len(la) == len(lb), f"index={a['index']} logprob length differs"
        for j, (xa, xb) in enumerate(zip(la, lb, strict=True)):
            assert abs(xa - xb) <= 1e-6, f"index={a['index']} logprob[{j}] {xa} vs {xb}"

        assert (
            a["rollout_routed_experts_shape"] == b["rollout_routed_experts_shape"]
        ), f"index={a['index']} routed_experts_shape mismatch"
        ea = a["rollout_routed_experts_b64"]
        eb = b["rollout_routed_experts_b64"]
        if ea is None and eb is None:
            continue
        assert ea is not None and eb is not None, f"index={a['index']} one side missing routed_experts"
        ba = base64.b64decode(ea)
        bb = base64.b64decode(eb)
        assert ba == bb, f"index={a['index']} routed_experts bytes differ"


def execute() -> None:
    cfg = _get_config()
    for variant in ("pd_off", "pd_on"):
        _run_variant(cfg, variant)

    off_records = _load_dump(_variant_dump_path("pd_off"))
    on_records = _load_dump(_variant_dump_path("pd_on"))

    assert off_records, "pd_off run produced no dump records"
    assert on_records, "pd_on run produced no dump records"

    _assert_records_equal(off_records, on_records)

    print(f"[pd-eq] model_family={MODEL_FAMILY} variants pd_off/pd_on " f"match across {len(off_records)} samples")


def test_pd_routing_expert_equivalence():
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
