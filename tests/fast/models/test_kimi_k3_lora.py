from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from scripts.run_kimi_k3_lora import _FULL_SGLANG_CONFIG, ScriptArgs, _execute_train
from torch.utils.checkpoint import checkpoint

from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles_plugins.models.kimi_k3.lora import (
    KimiK3LoRAAdapter,
    _enable_full_recompute_input_grads,
    _grouped_linear,
    _validate_targets,
    export_kimi_k3_lora_hf_chunks,
)


def _parameter(*shape):
    return nn.Parameter(torch.arange(torch.tensor(shape).prod()).reshape(shape).float())


def _mla_attention_adapter():
    adapter = KimiK3LoRAAdapter("mla_attention", "language_model.model.layers.3.self_attn.")
    adapter.register_parameter("q_a_lora_A", _parameter(2, 8))
    adapter.register_parameter("q_a_lora_B", _parameter(4, 2))
    adapter.register_parameter("kv_a_lora_A", _parameter(2, 8))
    adapter.register_parameter("kv_a_lora_B", _parameter(6, 2))
    adapter.register_parameter("o_lora_A", _parameter(2, 3))
    adapter.register_parameter("o_lora_B", _parameter(8, 2))
    return adapter


def _expert_adapter():
    adapter = KimiK3LoRAAdapter(
        "experts",
        "language_model.model.layers.4.block_sparse_moe.experts.",
    )
    adapter.register_parameter("w1_lora_A", _parameter(2, 8))
    adapter.register_parameter("w3_lora_A", _parameter(2, 8))
    adapter.register_parameter("w1_lora_B", _parameter(3, 5, 2))
    adapter.register_parameter("w3_lora_B", _parameter(3, 5, 2))
    adapter.register_parameter("w2_lora_A", _parameter(3, 2, 5))
    adapter.register_parameter("w2_lora_B", _parameter(8, 2))
    return adapter


def _shared_expert_adapter():
    adapter = KimiK3LoRAAdapter(
        "shared_experts",
        "language_model.model.layers.4.block_sparse_moe.shared_experts.",
    )
    adapter.register_parameter("fc1_lora_A", _parameter(2, 8))
    adapter.register_parameter("fc1_lora_B", _parameter(10, 2))
    adapter.register_parameter("fc2_lora_A", _parameter(2, 5))
    adapter.register_parameter("fc2_lora_B", _parameter(8, 2))
    return adapter


def _dense_adapter():
    adapter = KimiK3LoRAAdapter(
        "dense_mlp",
        "language_model.model.layers.3.mlp.",
    )
    adapter.register_parameter("fc1_lora_A", _parameter(2, 8))
    adapter.register_parameter("fc1_lora_B", _parameter(10, 2))
    adapter.register_parameter("fc2_lora_A", _parameter(2, 5))
    adapter.register_parameter("fc2_lora_B", _parameter(8, 2))
    return adapter


def _kda_attention_adapter():
    adapter = KimiK3LoRAAdapter(
        "kda_attention",
        "language_model.model.layers.4.self_attn.",
    )
    adapter.register_parameter("o_lora_A", _parameter(2, 3))
    adapter.register_parameter("o_lora_B", _parameter(8, 2))
    return adapter


def _model_with_adapters(*, include_shared_experts=True):
    mla_attention = _mla_attention_adapter()
    dense = _dense_adapter()
    kda_attention = _kda_attention_adapter()
    experts = _expert_adapter()
    shared_experts = _shared_expert_adapter()
    adapters = [mla_attention, dense, kda_attention, experts]
    if include_shared_experts:
        adapters.append(shared_experts)

    model = nn.Module()
    model.adapters = nn.ModuleList(adapters)
    model.decoder = SimpleNamespace(
        layers=[
            SimpleNamespace(
                layer_number=4,
                self_attention=SimpleNamespace(is_kda=False),
                mlp=SimpleNamespace(),
            ),
            SimpleNamespace(
                layer_number=5,
                self_attention=SimpleNamespace(is_kda=True),
                mlp=SimpleNamespace(
                    experts=SimpleNamespace(),
                    shared_experts=SimpleNamespace(),
                ),
            ),
        ]
    )
    return model


def test_native_export_is_chunked_by_adapter(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 1)

    model = _model_with_adapters()
    chunks = list(export_kimi_k3_lora_hf_chunks([model]))

    assert len(chunks) == 5
    attention = dict(chunks[0])
    assert attention["language_model.model.layers.3.self_attn.q_a_proj.lora_A.weight"].shape == (2, 8)
    assert attention["language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.lora_B.weight"].shape == (
        6,
        2,
    )
    assert attention["language_model.model.layers.3.self_attn.o_proj.lora_A.weight"].shape == (2, 3)

    experts = dict(chunks[3])
    prefix = "language_model.model.layers.4.block_sparse_moe.experts."
    assert experts[f"{prefix}w1.lora_A.weight"].shape == (1, 2, 8)
    assert experts[f"{prefix}w1.lora_B.weight"].shape == (3, 5, 2)
    assert experts[f"{prefix}w2.lora_A.weight"].shape == (3, 2, 5)
    assert experts[f"{prefix}w2.lora_B.weight"].shape == (1, 8, 2)

    shared_experts = dict(chunks[4])
    prefix = "language_model.model.layers.4.block_sparse_moe.shared_experts."
    assert shared_experts[f"{prefix}gate_proj.lora_A.weight"].shape == (2, 8)
    assert shared_experts[f"{prefix}gate_proj.lora_B.weight"].shape == (5, 2)
    assert shared_experts[f"{prefix}up_proj.lora_A.weight"].shape == (2, 8)
    assert shared_experts[f"{prefix}up_proj.lora_B.weight"].shape == (5, 2)
    assert shared_experts[f"{prefix}down_proj.lora_A.weight"].shape == (2, 5)
    assert shared_experts[f"{prefix}down_proj.lora_B.weight"].shape == (8, 2)


def test_native_export_materializes_from_backup(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 1)

    model = _model_with_adapters()
    backups = {id(parameter): torch.full_like(parameter, 17) for parameter in model.parameters()}
    chunks = list(
        export_kimi_k3_lora_hf_chunks([model], materialize_parameter=lambda parameter: backups[id(parameter)])
    )

    for chunk in chunks:
        for _name, tensor in chunk:
            torch.testing.assert_close(tensor, torch.full_like(tensor, 17))


def test_native_export_rejects_missing_shared_expert_adapter(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 1)

    model = _model_with_adapters(include_shared_experts=False)

    with pytest.raises(RuntimeError, match="adapter layout is incomplete"):
        list(export_kimi_k3_lora_hf_chunks([model]))


def test_grouped_linear_uses_expert_token_boundaries():
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])

    output = _grouped_linear(inputs, weights, [1, 2])

    torch.testing.assert_close(output, torch.tensor([[1.0], [4.0], [6.0]]))


def test_native_lora_rejects_target_drift():
    args = SimpleNamespace(
        target_modules=["decoder.layers.*.self_attention.o_proj"],
        experts_shared_outer_loras=True,
    )

    with pytest.raises(NotImplementedError, match="missing="):
        _validate_targets(args)


def test_full_recompute_keeps_native_lora_in_autograd_graph():
    model = nn.Module()
    model.embedding = nn.Embedding.from_pretrained(torch.ones(8, 4), freeze=True)
    model.adapter = nn.Parameter(torch.ones(4, 4))
    model.config = SimpleNamespace(recompute_granularity="full")
    model.pre_process = True
    model.embedding.requires_grad_(False)
    _enable_full_recompute_input_grads(model)

    model.train()
    hidden_states = model.embedding(torch.tensor([[1, 2]]))
    output = checkpoint(lambda inputs: inputs @ model.adapter, hidden_states, use_reentrant=True)
    output.sum().backward()

    assert hidden_states.requires_grad
    assert model.embedding.weight.grad is None
    torch.testing.assert_close(model.adapter.grad, torch.full_like(model.adapter, 2.0))

    model.eval()
    assert not model.embedding(torch.tensor([[1, 2]])).requires_grad


def test_full_rollout_uses_two_official_tp8_engines():
    config = SglangConfig.from_yaml(_FULL_SGLANG_CONFIG)

    assert config.total_num_gpus == 64
    assert config.models[0].num_gpus_per_engine == 8
    assert [(group.worker_type, group.num_gpus) for group in config.models[0].server_groups] == [
        ("regular", 16),
        ("placeholder", 48),
    ]


def test_full_rollout_matches_official_sglang_runtime(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr("scripts.run_kimi_k3_lora.U.execute_train", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr("scripts.run_kimi_k3_lora.U.get_default_wandb_args", lambda *args, **kwargs: "")
    hf_checkpoint = tmp_path / "hf"
    ref_load = tmp_path / "dcp"
    sglang_path = tmp_path / "sglang" / "python"
    data_dir = tmp_path / "data"
    dataset = data_dir / "dapo-math-17k" / "dapo-math-17k.jsonl"
    hf_checkpoint.mkdir()
    ref_load.mkdir()
    (sglang_path / "sglang").mkdir(parents=True)
    dataset.parent.mkdir(parents=True)
    dataset.touch()

    args = ScriptArgs(
        model_variant="full",
        task="dapo-math",
        num_nodes=16,
        num_gpus_per_node=4,
        checkpoint_load_mode="shared",
        hf_checkpoint=str(hf_checkpoint),
        ref_load=str(ref_load),
        megatron_model_type="kimi-k3",
        sglang_path=str(sglang_path),
        data_dir=str(data_dir),
    )
    _execute_train(args)

    train_args = captured["train_args"]
    for expected in (
        "--rollout-num-gpus-per-engine 8",
        "--sglang-tp-size 8",
        "--sglang-ep-size 1",
        "--sglang-moe-runner-backend marlin",
        "--sglang-decode-attention-backend trtllm_mla",
        "--sglang-mamba-radix-cache-strategy extra_buffer",
        "--sglang-cuda-graph-bs-decode 1",
        "--sglang-cuda-graph-backend-prefill disabled",
        "--sglang-lora-backend triton",
        "--sglang-lora-strict-loading",
    ):
        assert expected in train_args

    for unsupported in (
        "--sglang-dtype",
        "--sglang-quantization",
        "--sglang-mem-fraction-static",
        "--sglang-disable-shared-experts-fusion",
        "--sglang-weight-loader-drop-cache-after-load",
    ):
        assert unsupported not in train_args

    assert captured["extra_env_vars"]["SGLANG_JIT_ROUTE_RADIX"] == "1"
    assert "SGLANG_K3_ATTN_RES_MODE" not in captured["extra_env_vars"]
    assert "SGLANG_K3_FUSE_MOE_FRONT" not in captured["extra_env_vars"]


def test_full_gsm8k_uses_math_reward_and_messages(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr("scripts.run_kimi_k3_lora.U.execute_train", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr("scripts.run_kimi_k3_lora.U.get_default_wandb_args", lambda *args, **kwargs: "")
    hf_checkpoint = tmp_path / "hf"
    ref_load = tmp_path / "dcp"
    sglang_path = tmp_path / "sglang" / "python"
    data_dir = tmp_path / "data"
    dataset = data_dir / "gsm8k" / "train.parquet"
    hf_checkpoint.mkdir()
    ref_load.mkdir()
    (sglang_path / "sglang").mkdir(parents=True)
    dataset.parent.mkdir(parents=True)
    dataset.touch()

    args = ScriptArgs(
        model_variant="full",
        task="gsm8k",
        mode="normal",
        num_nodes=16,
        num_gpus_per_node=4,
        checkpoint_load_mode="shared",
        hf_checkpoint=str(hf_checkpoint),
        ref_load=str(ref_load),
        megatron_model_type="kimi-k3",
        sglang_path=str(sglang_path),
        data_dir=str(data_dir),
        sglang_max_total_tokens=8192,
        save_debug_rollout_data="/tmp/rollout-{rollout_id}.pt",
        enable_wandb=True,
    )
    _execute_train(args)

    train_args = captured["train_args"]
    for expected in (
        f"--prompt-data {dataset}",
        "--input-key messages",
        "--label-key label",
        "--rm-type math",
        "--rollout-max-response-len 256",
        "--sglang-max-total-tokens 8192",
        "--save-debug-rollout-data /tmp/rollout-{rollout_id}.pt",
        "--use-wandb",
        "--wandb-project miles-run_kimi_k3_lora",
    ):
        assert expected in train_args
