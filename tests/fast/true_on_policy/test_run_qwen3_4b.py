from __future__ import annotations

from scripts import run_qwen3_4b
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu")


def test_qwen3_script_true_on_policy_single_knob_expands_to_megatron_contract(monkeypatch):
    captured = {}

    def fake_execute_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_qwen3_4b.U, "execute_train", fake_execute_train)
    monkeypatch.setattr(run_qwen3_4b.U, "get_default_wandb_args", lambda *args, **kwargs: "")

    args = run_qwen3_4b.ScriptArgs(
        run_id="unit-test",
        model_name="Qwen3-4B",
        true_on_policy=True,
        enable_eval=False,
        use_kl_loss=False,
    )

    assert args.use_sequence_parallel is False

    run_qwen3_4b.execute(args)

    train_args = captured["train_args"]
    env_vars = captured["extra_env_vars"]

    assert "--true-on-policy-mode" in train_args
    assert "--sglang-enable-deterministic-inference" in train_args
    assert "--sglang-true-on-policy-contract qwen3_dense_true_on_policy_v1" in train_args
    assert "--sglang-rl-on-policy-target" not in train_args
    assert "--sglang-attention-backend fa3" in train_args
    assert "--true-on-policy-contract qwen3_dense_true_on_policy_v1" in train_args
    assert "--recompute-logprobs-via-prefill" in train_args
    assert "--load /root/models/Qwen3-4B_torch_dist" in train_args
    assert "--save /root/shared_data/unit-test/checkpoints" in train_args
    assert "--use-sglang" not in train_args
    assert "--batch-invariant-mode" in train_args
    assert "--no-rope-fusion" in train_args
    assert "--sequence-parallel" not in train_args
    assert "ROW_LINEAR_ENABLE_INV" not in env_vars
    assert "MEGATRON_USE_DETERMINISTIC_ALLREDUCE" not in env_vars


def test_qwen3_script_true_on_policy_tp2_cp4_normal_topology_contract(monkeypatch):
    captured = {}

    def fake_execute_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_qwen3_4b.U, "execute_train", fake_execute_train)
    monkeypatch.setattr(run_qwen3_4b.U, "get_default_wandb_args", lambda *args, **kwargs: "")

    args = run_qwen3_4b.ScriptArgs(
        run_id="unit-test-tp2-cp4",
        model_name="Qwen3-4B",
        mode="normal",
        true_on_policy=True,
        enable_eval=False,
        use_kl_loss=False,
    )
    args.tensor_model_parallel_size = 2
    args.context_parallel_size = 4
    args.pipeline_model_parallel_size = 1
    args.cp_comm_type = "a2a"
    args.rollout_num_gpus_per_engine = 8
    args.extra_args = "--rollout-num-gpus 8 "

    assert args.use_sequence_parallel is False

    run_qwen3_4b.execute(args)

    train_args = captured["train_args"]
    env_vars = captured["extra_env_vars"]

    assert "--tensor-model-parallel-size 2" in train_args
    assert "--context-parallel-size 4" in train_args
    assert "--cp-comm-type a2a" in train_args
    assert "--rollout-num-gpus 8" in train_args
    assert "--rollout-num-gpus-per-engine 8" in train_args
    assert "--load /root/models/Qwen3-4B_torch_dist" in train_args
    assert "--save /root/shared_data/unit-test-tp2-cp4/checkpoints" in train_args
    assert "--sglang-true-on-policy-contract qwen3_dense_true_on_policy_v1" in train_args
    assert "--sglang-rl-on-policy-target" not in train_args
    assert "--sglang-attention-backend fa3" in train_args
    assert "--recompute-logprobs-via-prefill" in train_args
    assert "--use-sglang" not in train_args
    assert "--batch-invariant-mode" in train_args
    assert "--no-bias-swiglu-fusion" in train_args
    assert "--no-rope-fusion" in train_args
    assert "--sequence-parallel" not in train_args
    assert env_vars["NCCL_ALGO"] == "Ring"
    assert env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
    assert env_vars["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "ROW_LINEAR_ENABLE_INV" not in env_vars
    assert "MEGATRON_USE_DETERMINISTIC_ALLREDUCE" not in env_vars


def test_qwen3_script_off_policy_does_not_emit_true_on_policy_contract(monkeypatch):
    captured = {}

    def fake_execute_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_qwen3_4b.U, "execute_train", fake_execute_train)
    monkeypatch.setattr(run_qwen3_4b.U, "get_default_wandb_args", lambda *args, **kwargs: "")

    args = run_qwen3_4b.ScriptArgs(
        run_id="unit-test",
        model_name="Qwen3-4B",
        true_on_policy=False,
        enable_eval=False,
        use_kl_loss=False,
    )

    run_qwen3_4b.execute(args)

    train_args = captured["train_args"]
    env_vars = captured["extra_env_vars"]

    assert "--true-on-policy-mode" not in train_args
    assert "--sglang-true-on-policy-contract" not in train_args
    assert "--sglang-rl-on-policy-target" not in train_args
    assert "--true-on-policy-contract" not in train_args
    assert "--recompute-logprobs-via-prefill" not in train_args
    assert "--use-sglang" not in train_args
    assert "ROW_LINEAR_ENABLE_INV" not in env_vars
