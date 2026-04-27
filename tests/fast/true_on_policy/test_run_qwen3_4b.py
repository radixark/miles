from __future__ import annotations

from scripts import run_qwen3_4b


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

    assert args.sglang_rl_on_policy_target == "fsdp_tp"
    assert args.use_sequence_parallel is False

    run_qwen3_4b.execute(args)

    train_args = captured["train_args"]
    env_vars = captured["extra_env_vars"]

    assert "--true-on-policy-mode" in train_args
    assert "--sglang-enable-deterministic-inference" in train_args
    assert "--sglang-rl-on-policy-target fsdp_tp" in train_args
    assert "--sglang-attention-backend fa3" in train_args
    assert "--use-sglang" in train_args
    assert "--batch-invariant-mode" in train_args
    assert "--no-rope-fusion" in train_args
    assert "--sequence-parallel" not in train_args
    assert env_vars["ROW_LINEAR_ENABLE_INV"] == "1"
    assert env_vars["MEGATRON_USE_DETERMINISTIC_ALLREDUCE"] == "1"


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
    assert "--sglang-rl-on-policy-target" not in train_args
    assert "--use-sglang" not in train_args
    assert "ROW_LINEAR_ENABLE_INV" not in env_vars
