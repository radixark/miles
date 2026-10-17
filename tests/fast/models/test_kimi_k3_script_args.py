from __future__ import annotations

import pytest

from scripts import run_kimi_k3


def _full(**kwargs):
    return run_kimi_k3.ScriptArgs(model_name="Kimi-K3", hardware="GB300", num_nodes=16, num_gpus_per_node=4, **kwargs)


def _four_layer(**kwargs):
    kwargs.setdefault("num_nodes", 1)
    return run_kimi_k3.ScriptArgs(model_name="Kimi-K3-4layer", hardware="H200", **kwargs)


# (pipeline, context, expected tp, expected ep) for every layout run in production.
@pytest.mark.parametrize(
    "pipeline_parallel_size,context_parallel_size,expected_tp,expected_ep",
    [(1, 1, 32, 64), (8, 1, 8, 8), (8, 2, 4, 8)],
)
def test_full_model_parallel_derivation(pipeline_parallel_size, context_parallel_size, expected_tp, expected_ep):
    args = _full(pipeline_parallel_size=pipeline_parallel_size, context_parallel_size=context_parallel_size)
    assert args.tensor_parallel_size == expected_tp
    assert args.expert_parallel_size == expected_ep


def test_derived_ep_saturates_the_bound_post_init_validates():
    """EP must fill the non-PP ranks of one stage; TP alone under-uses it whenever CP > 1."""
    args = _full(pipeline_parallel_size=8, context_parallel_size=2)
    model_parallel = args.tensor_parallel_size * args.context_parallel_size * args.pipeline_parallel_size
    data_parallel = 64 // model_parallel
    assert args.expert_parallel_size == args.tensor_parallel_size * args.context_parallel_size * data_parallel


def test_ep_override_wins_over_derivation():
    args = _full(pipeline_parallel_size=8, context_parallel_size=2, ep_size_override=4)
    assert args.expert_parallel_size == 4


def test_tp_override_wins_and_feeds_the_ep_derivation():
    args = _full(pipeline_parallel_size=4, context_parallel_size=1, tp_size_override=8)
    assert args.tensor_parallel_size == 8
    # DP is 2 here, so the stage still holds 16 non-PP ranks.
    assert args.expert_parallel_size == 16


def test_full_model_off_the_validated_gpu_count_needs_an_explicit_layout():
    """The 64-GPU derivation would silently produce a layout nobody has run."""
    with pytest.raises(ValueError, match="tp-size-override"):
        run_kimi_k3.ScriptArgs(model_name="Kimi-K3", hardware="H200", num_nodes=4, num_gpus_per_node=8)


@pytest.mark.parametrize("num_gpus_per_node,expected_tp", [(8, 8), (4, 4)], ids=["8-gpu-node", "4-gpu-node"])
def test_four_layer_layout_follows_the_gpu_count(num_gpus_per_node, expected_tp):
    """The rollout serves the experts replicated (EP1): Marlin is the only MXFP4 MoE runner with a LoRA path."""
    args = _four_layer(num_gpus_per_node=num_gpus_per_node)
    assert args.tensor_parallel_size == expected_tp
    assert args.expert_parallel_size == expected_tp
    assert args.rollout_tp_size == expected_tp
    assert args.rollout_ep_size == 1


def test_four_layer_rollout_tp_can_span_nodes():
    """TP16 pads the Marlin MoE intermediate, the layout the padding fix is only decidable on."""
    args = _four_layer(num_nodes=2, num_gpus_per_node=8, rollout_tp_size=16, rollout_ep_size=1)
    assert args.tensor_parallel_size == 8
    assert args.rollout_tp_size == 16
    assert args.rollout_ep_size == 1


def test_checkpoint_paths_derive_from_the_model_name():
    args = _four_layer(model_dir="/m")
    assert args.hf_checkpoint == "/m/Kimi-K3-4layer"
    assert args.bf16_checkpoint == "/m/Kimi-K3-4layer-bf16"
    assert args.ref_load == "/m/Kimi-K3-4layer-bf16_torch_dist"


class _CapturingBackend:
    def execute_train(self, **kwargs):
        self.kwargs = kwargs


def _launch(args, monkeypatch):
    backend = _CapturingBackend()
    monkeypatch.setattr(type(args), "create_backend", lambda _self: backend)
    run_kimi_k3._train(args)
    return backend.kwargs["train_args"].split(), backend.kwargs["extra_env_vars"]


def _full_lora(**kwargs):
    return _full(
        mode="normal",
        train_mode="lora",
        task="dapo-math",
        pipeline_parallel_size=8,
        context_parallel_size=2,
        rollout_tp_size=16,
        **kwargs,
    )


def test_defaults_launch_the_stock_offload_and_rollout_paths(monkeypatch):
    argv, env = _launch(_full_lora(), monkeypatch)

    assert "--no-offload-train-params" not in argv
    assert "--no-offload-train-process-groups" not in argv
    assert "--offload-rollout-level" not in argv
    assert argv[argv.index("--kimi-k3-situ-backend") + 1] == "eager"
    assert "NCCL_NVLS_ENABLE" not in env
    assert not any(key in env for key in run_kimi_k3._FAST_LORA_DECODE_ENV)
    assert "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2" not in env


def test_resident_colocate_keeps_both_sides_on_gpu(monkeypatch):
    argv, env = _launch(_full_lora(resident_colocate=True), monkeypatch)

    assert "--no-offload-train-params" in argv
    assert "--no-offload-train-process-groups" in argv
    level = argv.index("--offload-rollout-level")
    assert argv[level + 1] == "kv_cache" and argv[level + 2].startswith("--")  # weights stay with the engine
    assert argv[argv.index("--sglang-mem-fraction-static") + 1] == "0.88"
    assert env["NCCL_NVLS_ENABLE"] == "0"


def test_fast_lora_decode_sets_every_sglang_flag(monkeypatch):
    argv, env = _launch(_full_lora(fast_lora_decode=True, situ_backend="fused"), monkeypatch)

    assert {key: env[key] for key in run_kimi_k3._FAST_LORA_DECODE_ENV} == run_kimi_k3._FAST_LORA_DECODE_ENV
    assert env["SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"] == "1"
    assert argv[argv.index("--kimi-k3-situ-backend") + 1] == "fused"


@pytest.mark.parametrize("option", ["resident_colocate", "fast_lora_decode"])
def test_lora_only_options_are_refused_for_full_training(option):
    with pytest.raises(ValueError, match="LoRA-only"):
        _full(train_mode="full", pipeline_parallel_size=8, context_parallel_size=2, **{option: True})


def test_fast_lora_decode_refuses_the_streamed_adapter_checksum():
    with pytest.raises(ValueError, match="check_lora_weight_equal"):
        _full_lora(fast_lora_decode=True, check_lora_weight_equal=True)
