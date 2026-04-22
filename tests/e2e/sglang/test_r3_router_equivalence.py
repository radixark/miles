"""E2E test: verify sglang router and miles router produce identical rollout
routing replay results across MoE models.

Design
~~~~~~
For each model in ``MODEL_REGISTRY``, run the same rollout workload twice
under ``--debug-rollout-only --sglang-enable-deterministic-inference
--use-rollout-routing-replay``:

1. ``variant=miles``: with ``--use-miles-router`` (Python middleware
   router wrapping the Rust gateway).
2. ``variant=sgl``: without ``--use-miles-router`` (direct Rust gateway,
   which is what PR #1015 drops R3 tests onto).

Each run writes a JSONL of per-sample ``(tokens, rollout_log_probs,
rollout_routed_experts)`` via the custom generate function in
``utils.router_equivalence_generate``.  Once both runs finish we diff
the dumps; they must match byte-for-byte (deterministic inference +
identical prompts).

Backend / checkpoint
~~~~~~~~~~~~~~~~~~~~
Megatron backend (same as the sibling ``tests/e2e/megatron/*_r3.py``
tests) — sourcing ``scripts/models/{type}.sh`` populates
``args.num_layers`` / ``args.moe_router_topk`` that the rollout-side
reshape of ``routed_experts`` depends on.  We do *not* set
``--use-kl-loss`` or ``--kl-coef`` > 0, which is what gates the
``--ref-load`` existence check (``miles/utils/arguments.py``), and
``--debug-rollout-only`` makes ``_compute_megatron_num_gpus`` return
``0`` so no megatron actor is spawned and the checkpoint is never
loaded.  This lets us get away with a single H200 and no
``convert_hf_to_torch_dist`` step.

Controls
~~~~~~~~
- ``ROUTER_EQ_MODEL_FAMILY``: ``qwen3_30b_a3b`` (default) | ``glm47_flash``.
- Single H200, bf16, rollout batch 10, num_rollout 1. Single engine
  (``--rollout-num-gpus-per-engine 1``) so both variants hit the same
  underlying sglang process topology.
"""

import base64
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import miles.utils.external_utils.command_utils as U

MODEL_FAMILY = os.environ.get("ROUTER_EQ_MODEL_FAMILY", "qwen3_30b_a3b")
DUMP_ROOT = Path(os.environ.get("ROUTER_EQ_DUMP_ROOT", "/tmp/router-eq"))
PROMPT_DATA_PATH = "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl"
NUM_PROMPTS = int(os.environ.get("ROUTER_EQ_NUM_PROMPTS", "10"))
MAX_RESPONSE_LEN = int(os.environ.get("ROUTER_EQ_MAX_RESPONSE_LEN", "256"))

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
    num_gpus: int = 1


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "qwen3_30b_a3b": ModelConfig(
        model_name="Qwen3-30B-A3B",
        hf_repo="Qwen/Qwen3-30B-A3B",
        local_dir="/root/models/Qwen3-30B-A3B",
        megatron_model_type="qwen3-30B-A3B",
        reasoning_parser=None,
        num_gpus=1,
    ),
    "glm47_flash": ModelConfig(
        model_name="GLM-4.7-Flash",
        hf_repo="zai-org/GLM-4.7-Flash",
        local_dir="/root/models/GLM-4.7-Flash",
        megatron_model_type="glm4.7-flash",
        reasoning_parser="glm45",
        num_gpus=1,
    ),
}


def _get_config() -> ModelConfig:
    if MODEL_FAMILY not in MODEL_REGISTRY:
        raise ValueError(f"Unknown ROUTER_EQ_MODEL_FAMILY={MODEL_FAMILY!r}; choose from {list(MODEL_REGISTRY)}")
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


def _build_train_args(cfg: ModelConfig, variant: str) -> str:
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
    if variant == "miles":
        router_args += "--use-miles-router "

    # Minimal megatron perf args — 1 GPU, no parallelism. We don't actually
    # start a megatron actor under --debug-rollout-only, so these are only
    # consumed by the argument parser.
    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {cfg.num_gpus} "
        "--sglang-enable-deterministic-inference "
        "--sglang-mem-fraction-static 0.85 "
    )
    if cfg.reasoning_parser:
        sglang_args += f"--sglang-reasoning-parser {cfg.reasoning_parser} "

    infra_args = (
        "--debug-rollout-only "
        "--ci-test "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {cfg.num_gpus} "
        "--colocate "
    )

    return ckpt_args + rollout_args + generate_args + router_args + perf_args + sglang_args + infra_args


def _run_variant(cfg: ModelConfig, variant: str) -> None:
    dump_dir = _variant_dir(variant)
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = _variant_dump_path(variant)

    train_args = _build_train_args(cfg, variant)
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=cfg.num_gpus,
        megatron_model_type=cfg.megatron_model_type,
        extra_env_vars={
            "PYTHONPATH": "/root/Megatron-LM",
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

        # Tokens and status must match exactly — deterministic decoding.
        for field in ("status", "response_length", "tokens"):
            assert (
                a[field] == b[field]
            ), f"index={a['index']} field={field} mismatch:\n  miles: {a[field]}\n  sgl:   {b[field]}"

        # Logprobs are f32 from sglang; in deterministic mode they should be
        # bit-identical, but tolerate tiny float noise as a safety margin.
        la = a["rollout_log_probs"] or []
        lb = b["rollout_log_probs"] or []
        assert len(la) == len(lb), f"index={a['index']} logprob length differs"
        for j, (xa, xb) in enumerate(zip(la, lb, strict=True)):
            assert abs(xa - xb) <= 1e-6, f"index={a['index']} logprob[{j}] {xa} vs {xb}"

        # routed_experts: deterministic int32 → must be byte-identical.
        assert (
            a["rollout_routed_experts_shape"] == b["rollout_routed_experts_shape"]
        ), f"index={a['index']} routed_experts_shape mismatch"
        ea = a["rollout_routed_experts_b64"]
        eb = b["rollout_routed_experts_b64"]
        if ea is None and eb is None:
            continue
        assert ea is not None and eb is not None, f"index={a['index']} one side missing routed_experts"
        # Compare raw bytes, not the base64 string (equivalent, but clearer error).
        ba = base64.b64decode(ea)
        bb = base64.b64decode(eb)
        assert ba == bb, f"index={a['index']} routed_experts bytes differ"


def execute() -> None:
    cfg = _get_config()
    for variant in ("miles", "sgl"):
        _run_variant(cfg, variant)

    miles_records = _load_dump(_variant_dump_path("miles"))
    sgl_records = _load_dump(_variant_dump_path("sgl"))

    assert miles_records, "miles-router run produced no dump records"
    assert sgl_records, "sglang-router run produced no dump records"

    _assert_records_equal(miles_records, sgl_records)

    print(f"[router-eq] model_family={MODEL_FAMILY} variants miles/sgl " f"match across {len(miles_records)} samples")


def test_r3_router_equivalence():
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
