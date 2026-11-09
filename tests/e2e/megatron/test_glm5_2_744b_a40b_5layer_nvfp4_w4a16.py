import json
import os
import shlex
from pathlib import Path

from tests.ci.ci_register import register_cuda_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(
    # Conservative TP2/DCP2 bound; registered-runner timing is pending.
    est_time=2400,
    suite="stage-c-8-gpu-b200",
    labels=["megatron", "model-scripts"],
    hardware=["blackwell"],
    num_gpus=8,
)

MODEL_ORG = "Pinaster"
MODEL_NAME = "GLM-5.2_5layer"
MODEL_TYPE = "glm5.2-744B-A40B_5layer"
MEGATRON_MODEL_NAME = f"{MODEL_NAME}-megatron-dsa"
NVFP4_MODEL_NAME = f"{MODEL_NAME}-NVFP4-w4a16"
DSA_ARGS = "--megatron-to-hf-mode raw --dsa-impl megatron --dsa-kernel-backend cudnn "
NUM_GPUS = 8
ACTOR_NUM_GPUS = 4
ROLLOUT_NUM_GPUS = 4
ROLLOUT_GPUS_PER_ENGINE = 2
RUN_ID = command_utils.create_run_id()

MODEL_DIR = "/root/models"
DATA_DIR = "/root/datasets"
MEGATRON_PATH = "/root/TransformerEngine:/root/Megatron-LM"

EXTRA_HIGH_PRECISION_LAYERS_HF = (".shared_experts.",)
EXTRA_HIGH_PRECISION_LAYERS_MEGATRON = (
    ".shared_experts.linear_fc1",
    ".shared_experts.linear_fc2",
)

NVFP4_ENV = {
    "OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG": "1",
    "SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16": "1",
    "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE": "0",
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_USE_FAST_MATH": "0",
    "NVTE_NVFP4_4OVER6": "all",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "none",
    "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
    "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
}

GLM5_ENV = {
    "SGLANG_DSA_FUSE_TOPK": "1",
    "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
    "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "large",
    "INDEXER_ROPE_NEOX_STYLE": "0",
    "NVSHMEM_DISABLE_NCCL": "1",
}

# Fake QDQ supplies NVFP4 expert weights; TE runs BF16 forward and backward GEMMs.
TE_PRECISION_CONFIG = """
configs:
    bf16:
        transformer_engine_config_type: "TEQuantizationParams"
        training_recipe: {}
matchers:
    default_bf16:
        type: "glob"
        enabled: true
        pattern: "*"
        config: "bf16"
""".strip()


def _extra_high_precision_layers_hf_args() -> str:
    return "--extra-high-precision-layers-hf " + " ".join(EXTRA_HIGH_PRECISION_LAYERS_HF) + " "


def _extra_high_precision_layers_megatron_args() -> str:
    return "--extra-high-precision-layers-megatron " + " ".join(EXTRA_HIGH_PRECISION_LAYERS_MEGATRON) + " "


def _validate_glm_checkpoint():
    config_path = Path(MODEL_DIR) / MODEL_NAME / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found")

    with open(config_path) as f:
        config = json.load(f)

    if (
        config.get("model_type") != "glm_moe_dsa"
        or config.get("architectures") != ["GlmMoeDsaForCausalLM"]
        or config.get("num_hidden_layers") != 5
    ):
        raise RuntimeError(
            f"{config_path} must use native GLM-5.2 5-layer config with "
            f"model_type=glm_moe_dsa, architectures=[GlmMoeDsaForCausalLM], "
            "and num_hidden_layers=5"
        )
    if "auto_map" in config:
        raise RuntimeError(f"{config_path} must not contain auto_map. Try updating the checkpoint.")


def prepare():
    U = command_utils.default_config().create_backend()
    os.environ.update(NVFP4_ENV)
    U.exec_command_cpu(f"mkdir -p {MODEL_DIR} {DATA_DIR}")
    U.exec_command_cpu(f"hf download {MODEL_ORG}/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=DATA_DIR)

    _validate_glm_checkpoint()
    U.exec_command_gpu_once(
        f"python tools/convert_hf_to_nvfp4.py "
        f"--model-dir {MODEL_DIR}/{MODEL_NAME} "
        f"--save-dir {MODEL_DIR}/{NVFP4_MODEL_NAME} "
        f"{_extra_high_precision_layers_hf_args()}",
        f"{MODEL_DIR}/{NVFP4_MODEL_NAME}",
    )

    U.convert_checkpoint(
        model_name=MEGATRON_MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=1,
        extra_args=(
            f"{DSA_ARGS}"
            "--tensor-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--pipeline-model-parallel-size 1 "
            "--expert-model-parallel-size 1 "
        ),
        dir_dst=MODEL_DIR,
        hf_checkpoint=f"{MODEL_DIR}/{MODEL_NAME}",
        megatron_path=MEGATRON_PATH,
    )


def _gpu_delta_env():
    # Ray jobs receive an explicit environment, not every variable in this shell.
    return {
        "GPU_DELTA_SKIP_PAYLOAD_HASH": os.environ.get("GPU_DELTA_SKIP_PAYLOAD_HASH", "0"),
        "SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN": "0",
    }


def _install_gpu_delta_performance_check(args):
    from miles.backends.training_utils.weight_update.protocols.gpu_delta import session
    from miles.utils.gpu_delta.publication import configured_codec

    activate = session.activate_publication
    ordinary_codec = configured_codec()
    seen, engines, updates = set(), {}, []
    encoding_seen, encoding_updates = set(), {}
    report_path = Path(args.update_weight_disk_dir) / "ci-performance.json"

    async def checked_activate(clients, cohort, publication, session_id=None, weight_version=None, **kwargs):
        result = await activate(clients, cohort, publication, session_id, weight_version, **kwargs)
        expected = {identity["rank_id"] for group in cohort.participants for identity in group}
        receipts = result["resumed_receipts"]
        assert len(receipts) == len(expected) and {r["identity"]["rank_id"] for r in receipts} == expected
        codec, version = publication["codec"], publication["target_version"]
        for group in cohort.participants:
            assert sorted(identity["tp_rank"] for identity in group) == list(range(ROLLOUT_GPUS_PER_ENGINE))
            assert all(identity["dp_rank"] == 0 for identity in group)
            engines[group[0]["engine_id"]] = {"ranks": [r["rank_id"] for r in group], "version": version}
        warm = []
        for receipt in receipts:
            rank = receipt["identity"]["rank_id"]
            timing = receipt["result"]["timings"]
            # No-op publications do not warm the compressed path or satisfy its performance gate.
            if timing["compressed_batches"] > 0:
                key = rank, codec
                if key in seen and version > 1:
                    scheduler = receipt["scheduler_timing"]
                    warm.append(
                        {
                            "rank": rank,
                            "prepare_s": timing["host_prepare_s"],
                            "pause_s": (scheduler["resumed_ns"] - scheduler["pause_started_ns"]) / 1e9,
                            "apply_kernel_cache_misses": timing["apply_kernel_cache_misses"],
                        }
                    )
                seen.add(key)
        if warm:
            warm_ranks = {rank["rank"] for rank in warm}
            warm_engines = [
                {
                    "engine": timing["engine_id"],
                    # Adjacent tokenizer spans include all-TP acknowledgement before admission reopens.
                    "apply_and_resume_s": timing["apply_s"] + timing["resume_s"],
                }
                for timing in result["engine_timings"]
                if set(engines[timing["engine_id"]]["ranks"]).issubset(warm_ranks)
            ]
            updates.append({"version": version, "codec": codec, "ranks": warm, "engines": warm_engines})
        if (
            codec == ordinary_codec
            and version > 1
            and version not in encoding_updates
            and "producer_metrics" in publication
        ):
            owners = []
            for owner in publication["producer_metrics"]:
                key = owner["owner_rank"], codec
                nonempty = (
                    owner["encoder_batches"] > 0 and owner["changed_bytes"] > 0 and owner["matrix_payload_bytes"] > 0
                )
                owners.append(
                    {
                        "owner_rank": owner["owner_rank"],
                        "nonempty": nonempty,
                        "warm": nonempty and key in encoding_seen,
                        # Sequential measured stages, not pure compressor time. Includes
                        # H2D/XOR, metadata waits (including export dependencies), and final D2H.
                        "encoding_stage_wall_s": owner["encode_tensor_wall_sum_s"] + owner["finalize_wall_s"],
                    }
                )
                if nonempty:
                    encoding_seen.add(key)
            encoding_updates[version] = {"codec": codec, "owners": owners}
        # This runs after every engine in the cohort resumed, outside measured prepare/pause.
        report_path.write_text(
            json.dumps(
                {
                    "last_version": args.num_rollout,
                    "engines": engines,
                    "ordinary_codec": ordinary_codec,
                    "updates": updates,
                    "encoding_updates": encoding_updates,
                },
                allow_nan=False,
            )
        )
        return result

    # The protocol resolves this module function at activation time. Install only in this CI trainer.
    session.activate_publication = checked_activate


def _assert_gpu_delta_performance(report_path):
    report = json.loads(Path(report_path).read_text())
    engines = report["engines"]
    assert len(engines) == ROLLOUT_NUM_GPUS // ROLLOUT_GPUS_PER_ENGINE
    current_ranks = set()
    for engine in engines.values():
        assert engine["version"] == report["last_version"]
        assert len(engine["ranks"]) == ROLLOUT_GPUS_PER_ENGINE
        current_ranks.update(engine["ranks"])
    assert len(current_ranks) == ROLLOUT_NUM_GPUS
    missing = current_ranks - {
        rank["rank"]
        for update in report["updates"]
        if update["codec"] == report["ordinary_codec"]
        for rank in update["ranks"]
    }
    assert not missing, f"GPU-delta CI has no nonempty warm ordinary-codec update for {len(missing)} current ranks"
    warm = [rank for update in report["updates"] for rank in update["ranks"]]
    assert warm, "GPU-delta CI did not validate any nonempty warm update"
    assert all(
        rank["apply_kernel_cache_misses"] == 0 for rank in warm
    ), "GPU-delta CI warm updates must reuse the prepared Triton kernels"
    # Provisional five-layer CI limits: allow CPU headroom over the full-model <5s prepare proxy.
    for name, limit in (("prepare_s", 6.0), ("pause_s", 0.3)):
        failures = [rank[name] for rank in warm if not 0 <= rank[name] < limit]
        assert not failures, f"GPU-delta CI warm {name} must be <{limit}s; violations={failures}"
    warm_engines = [engine for update in report["updates"] for engine in update["engines"]]
    assert set(engines) == {
        engine["engine"] for engine in warm_engines
    }, "GPU-delta CI requires warm all-TP apply/resume timing for every engine"
    failures = [engine["apply_and_resume_s"] for engine in warm_engines if not 0 <= engine["apply_and_resume_s"] < 0.3]
    assert not failures, f"GPU-delta CI warm engine apply/resume must be <0.3s; violations={failures}"
    owners = [owner for update in report["encoding_updates"].values() for owner in update["owners"]]
    nonempty_owners = {owner["owner_rank"] for owner in owners if owner["nonempty"]}
    warm_owners = [owner for owner in owners if owner["warm"]]
    assert nonempty_owners and nonempty_owners == {
        owner["owner_rank"] for owner in warm_owners
    }, "GPU-delta CI requires nonempty warm ordinary encoding for every matrix-producing owner"
    # Provisional budget from full-model measured encoding stages, not pure compression.
    # Leave CPU headroom until timings are calibrated on the registered CI runner.
    encoding_limit_s = 6.0
    failures = [
        owner["encoding_stage_wall_s"]
        for owner in warm_owners
        if not 0 <= owner["encoding_stage_wall_s"] < encoding_limit_s
    ]
    assert not failures, f"GPU-delta CI warm encoding_stage_wall_s must be <{encoding_limit_s}s; violations={failures}"
    print(
        f"GPU-delta CI warm ranks={len(current_ranks)} checks={len(warm)} "
        f"prepare_max_s={max(rank['prepare_s'] for rank in warm)} "
        f"scheduler_pause_max_s={max(rank['pause_s'] for rank in warm)} "
        f"engine_apply_and_resume_max_s={max(engine['apply_and_resume_s'] for engine in warm_engines)} "
        f"encoding_owners={len(nonempty_owners)} "
        f"encoding_stage_wall_max_s={max(owner['encoding_stage_wall_s'] for owner in warm_owners)}",
        flush=True,
    )


def _assert_gpu_delta_weights_changed(args, version_dir, _rollout_engines):
    """Reject a successful-looking E2E whose learned publications were all no-ops."""
    import torch.distributed as dist

    from miles.utils.gpu_delta.publication import configured_codec

    if dist.get_rank() != 0:
        return
    version_dir = Path(version_dir)
    current = json.loads((version_dir / "manifest.json").read_text())
    if current["target_version"] == 1:
        _install_gpu_delta_performance_check(args)
    # This E2E publishes once at startup and between rollouts, independently of optimizer steps.
    startup_publications = 1
    last_version = startup_publications + args.num_rollout - 1
    if current["target_version"] != last_version:
        return
    initial_codec = configured_codec(initial_sync=True)
    ordinary_codec = configured_codec()
    changed_bytes = []
    for version in range(1, last_version + 1):
        manifest = json.loads((version_dir.parent / f"weight_v{version:06d}/manifest.json").read_text())
        assert manifest["stream_id"] == current["stream_id"]
        assert manifest["base_version"] == version - 1 and manifest["target_version"] == version
        expected_codec = initial_codec if version <= startup_publications else ordinary_codec
        assert manifest["codec"] == expected_codec, "E2E publication codec differs from configured phase codec"
        raw_tensors = [tensor for tensor in manifest["tensors"] if len(tensor["shape"]) <= 1]
        assert raw_tensors, "E2E must exercise direct scalar/vector targets"
        for tensor in manifest["tensors"]:
            direct = len(tensor["shape"]) <= 1
            assert tensor["encoding"] == (
                "raw_bytes" if direct else "xor_bytes"
            ), "E2E tensor encoding differs from its shape"
            if direct:
                assert not tensor["frames"] and "outer" not in tensor
                assert tensor.get("raw", {}).get("encoded_bytes", 0) == (
                    tensor["nbytes"] if tensor["changed_bytes"] else 0
                ), "E2E direct target is incomplete"
        if version > startup_publications:
            changed_bytes.append(sum(tensor["changed_bytes"] for tensor in manifest["tensors"]))
    assert any(count > 0 for count in changed_bytes), (
        f"GPU-delta E2E produced only no-op learned publications: {changed_bytes}. "
        "Version changes alone do not exercise a learned weight delta."
    )
    print(f"GPU-delta E2E learned publication changed bytes: {changed_bytes}", flush=True)


def execute():
    U = command_utils.default_config().create_backend()
    weight_transfer_args = (
        "--update-weight-transfer-mode gpu-delta "
        f"--update-weight-disk-dir /root/delta-updates/{RUN_ID}/gpu_delta "
        "--custom-update-weight-post-write-path "
        "tests.e2e.megatron.test_glm5_2_744b_a40b_5layer_nvfp4_w4a16._assert_gpu_delta_weights_changed "
    )
    delta_env = _gpu_delta_env()

    os.environ.update(NVFP4_ENV)
    os.environ.update(GLM5_ENV)
    os.environ.update(delta_env)
    os.environ.setdefault("RAY_TMPDIR", "/tmp/ray")
    te_precision_config_path = command_utils.encode_pseudo_file(TE_PRECISION_CONFIG)

    ckpt_args = (
        f"--hf-checkpoint {MODEL_DIR}/{NVFP4_MODEL_NAME}/ "
        f"--ref-load {MODEL_DIR}/{MEGATRON_MODEL_NAME}_torch_dist "
        f"--save /root/shared_data/{RUN_ID}/checkpoints --save-interval 2 "
    )

    # Exercise learned deltas despite this pruned model's zero math score.
    rollout_args = (
        f"--prompt-data {DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deterministic_random "
        # Exercise repeated warm ordinary-codec updates on every TP rank.
        "--num-rollout 5 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 100 "
        "--rollout-temperature 1 "
        "--global-batch-size 16 "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        # Native AbsorbedMLA requires sequence parallelism with tensor parallelism.
        "--sequence-parallel "
        # Let the STE propagate gradients to the original expert parameters.
        "--no-gradient-accumulation-fusion "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        f"--expert-model-parallel-size {ACTOR_NUM_GPUS} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
        "--data-pad-size-multiplier 1024 "
        "--log-probs-chunk-size 16384 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-tis "
        "--tis-clip-low 0.5 "
        "--tis-clip 2.0 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-attention-backend nsa "
        "--sglang-dsa-decode-backend trtllm "
        "--sglang-dsa-prefill-backend trtllm "
        "--sglang-dsa-topk-backend flashinfer "
        "--sglang-kv-cache-dtype fp8_e4m3 "
        "--sglang-page-size 64 "
        f"--rollout-num-gpus-per-engine {ROLLOUT_GPUS_PER_ENGINE} "
        "--sglang-moe-runner-backend flashinfer_cutedsl "
        "--sglang-moe-a2a-backend none "
        # Each engine shards attention, routed experts, and BF16 shared experts across TP2.
        "--sglang-ep-size 1 "
        "--sglang-dp-size 1 "
        f"--sglang-moe-dense-tp-size {ROLLOUT_GPUS_PER_ENGINE} "
        f"--sglang-dcp-size {ROLLOUT_GPUS_PER_ENGINE} "
        # Pin the default DCP Q replica so updates also cover its derived weights.
        "--sglang-dcp-replicate-q-proj "
        "--sglang-enable-fp32-lm-head "
        "--sglang-cuda-graph-max-bs-decode 256 "
        "--sglang-max-running-requests 512 "
        f"--sglang-chunked-prefill-size {2048 * ROLLOUT_GPUS_PER_ENGINE} "
        "--sglang-watchdog-timeout 3600 "
    )

    ci_args = "--ci-test --ci-disable-logprobs-checker --ci-disable-weight-update-checker --ci-disable-kl-checker "

    mixed_precision_args = (
        "--transformer-impl transformer_engine "
        "--bf16 "
        f"{_extra_high_precision_layers_hf_args()}"
        f"{_extra_high_precision_layers_megatron_args()}"
        f"--te-precision-config-file {te_precision_config_path} "
    )

    misc_args = (
        f"{DSA_ARGS}"
        "--qkv-format thd "
        "--use-rollout-routing-replay "
        "--use-miles-router "
        "--sglang-disable-shared-experts-fusion "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--moe-router-use-torch-mm "
        "--attention-backend flash "
        "--cp-comm-type allgather "
        "--miles-dsa-topk-backend flashinfer "
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {ACTOR_NUM_GPUS} "
        f"--num-gpus-per-node {NUM_GPUS} "
        f"--rollout-num-gpus {ROLLOUT_NUM_GPUS} "
        "--moe-enable-deepep "
        "--moe-token-dispatcher-type flex "
        # Event logging requests weight checksums that SGLang does not support for NVFP4.
        f"--save-debug-rollout-data /root/shared_data/{RUN_ID}/dump_details/rollout_data/{{rollout_id}}.pt "
        f"--save-debug-train-data /root/shared_data/{RUN_ID}/dump_details/train_data/{{rollout_id}}_{{rank}}.pt "
        f"--save-debug-trajectory-data /root/shared_data/{RUN_ID}/dump_details/trajectory/{{rollout_id}}.jsonl "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=RUN_ID)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{mixed_precision_args} "
        f"{misc_args} "
        f"{weight_transfer_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        megatron_path=MEGATRON_PATH,
        extra_env_vars={
            **NVFP4_ENV,
            **GLM5_ENV,
            **delta_env,
            # DCP2 doubles this to 1.5 GiB; FlashInfer autotune needs more than 1 GiB.
            "SGLANG_FLASHINFER_WORKSPACE_SIZE": "805306368",
        },
    )
    check = (
        "from tests.e2e.megatron.test_glm5_2_744b_a40b_5layer_nvfp4_w4a16 import _assert_gpu_delta_performance; "
        f"_assert_gpu_delta_performance('/root/delta-updates/{RUN_ID}/gpu_delta/ci-performance.json')"
    )
    U.exec_command_cpu(f"python -c {shlex.quote(check)}")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
