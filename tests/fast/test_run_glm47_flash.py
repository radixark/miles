import shlex

import pytest

from scripts import run_glm47_flash


@pytest.mark.parametrize(
    ("hardware", "sglang_attention_backend", "expected_rollout_tp", "expect_disable_ragged"),
    [
        ("H200", None, "1", False),
        ("H200", "default", "1", False),
        ("H200", "flashinfer", "1", False),
        ("H200", "triton", "1", False),
        ("B200", None, "2", True),
        ("B200", "default", "2", True),
        ("B200", "flashinfer", "2", True),
        ("B200", "triton", "2", False),
    ],
)
def test_sglang_args_match_hardware_and_attention_backend(
    monkeypatch,
    hardware,
    sglang_attention_backend,
    expected_rollout_tp,
    expect_disable_ragged,
):
    captured = {}
    monkeypatch.setattr(run_glm47_flash.U, "execute_train", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(run_glm47_flash.U, "get_default_wandb_args", lambda *args, **kwargs: "")

    args = run_glm47_flash.ScriptArgs(
        run_id="unit-test",
        hardware=hardware,
        sglang_attention_backend=sglang_attention_backend,
        enable_eval=False,
    )
    run_glm47_flash.execute(args)
    train_args = shlex.split(captured["train_args"])

    rollout_tp_index = train_args.index("--rollout-num-gpus-per-engine")
    assert train_args[rollout_tp_index + 1] == expected_rollout_tp
    assert ("--sglang-flashinfer-mla-disable-ragged" in train_args) is expect_disable_ragged

    if sglang_attention_backend in (None, "default"):
        assert "--sglang-attention-backend" not in train_args
    else:
        attention_backend_index = train_args.index("--sglang-attention-backend")
        assert train_args[attention_backend_index + 1] == sglang_attention_backend

    assert "--tensor-model-parallel-size" in train_args
    assert train_args[train_args.index("--tensor-model-parallel-size") + 1] == "4"
    assert train_args[train_args.index("--expert-model-parallel-size") + 1] == "8"
    assert train_args[train_args.index("--actor-num-gpus-per-node") + 1] == "8"
    assert train_args[train_args.index("--num-gpus-per-node") + 1] == "8"
