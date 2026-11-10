import shlex
import sys

import pytest
import torch
from tests.fast.launch_scripts.model_args_harness import expand_model_args
from tests.fast.launch_scripts.py_harness import import_launch_script

import miles.utils.external_utils.command_utils.legacy as U

_H200 = "scripts/run_muse_spark_64xh200.py"
_B300 = "scripts/run_muse_spark_32xb300.py"
# Entry point -> (num_nodes, rollout GPUs per engine, max tokens per GPU, mem fraction, running requests, graph bs).
_HARDWARE = {
    _H200: (8, 16, 4096, 0.8, 64, [1, 2, 4, 8, 16, 32, 64]),
    _B300: (4, 8, 8192, 0.83, 128, [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]),
}
_LAYOUT_FIELDS = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "decoder_first_last_pipeline_num_layers",
)


def _train_args(monkeypatch, launcher, **kwargs) -> tuple[list[str], dict]:
    calls = []
    monkeypatch.setattr(U, "exec_command_cpu", lambda command: None)
    monkeypatch.setattr(U, "convert_checkpoint", lambda **kw: None)
    monkeypatch.setattr(U, "hf_download_dataset", lambda *a, **kw: None)
    monkeypatch.setattr(U, "execute_train", lambda **kw: calls.append(kw))
    launcher.main(**kwargs)
    (call,) = calls
    return shlex.split(call["train_args"]), call


# Layouts as (TP, PP, CP, EP, layers on each edge stage); None keeps the entry point's own.
@pytest.mark.parametrize(
    "script,layout,optimizer",
    [
        (_H200, None, "adam"),
        (_H200, (4, 8, 2, 8, 7), "muon"),
        (_H200, (4, 4, 4, 16, 15), "muon"),
        (_B300, None, "adam"),
        (_B300, None, "muon"),
    ],
)
@pytest.mark.parametrize("counts", [(3, 8, 8), (100, 32, 16)])
def test_dapo_recipe_survives_megatron_validation(monkeypatch, tmp_path, counts, script, layout, optimizer):
    from miles.backends.megatron_utils.arguments import parse_args, set_default_megatron_args, validate_args
    from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    launcher = import_launch_script(U.repo_base_dir / script)
    template = tmp_path / "chat.jinja"
    template.write_text("{{ messages }}")
    defaults = launcher.ScriptArgs()
    if layout is None:
        layout = tuple(getattr(defaults, name) for name in _LAYOUT_FIELDS)
    use_r3 = script == _B300 or layout[2] > 1
    tokens, call = _train_args(
        monkeypatch,
        launcher,
        **dict(zip(_LAYOUT_FIELDS, layout, strict=True)),
        optimizer=optimizer,
        reasoning_effort="xhigh" if optimizer == "muon" else None,
        rollout_max_response_len=4096 if optimizer == "muon" else 2048,
        use_rollout_routing_replay=use_r3,
        num_rollout=counts[0],
        rollout_batch_size=counts[1],
        n_samples_per_prompt=counts[2],
        chat_template_path=str(template),
    )
    num_nodes, gpus_per_engine, max_tokens, mem_fraction, running, graph_bs = _HARDWARE[script]
    assert "--use-dynamic-batch-size" in tokens and "--micro-batch-size" not in tokens
    assert call["num_gpus_per_node"] == 8 and call["config"].num_nodes == num_nodes
    blackwell = script == _B300
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3) if blackwell else (9, 0))
    monkeypatch.setattr("megatron.training.arguments.get_device_arch_version", lambda: 10 if blackwell else 9)
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", *expand_model_args("muse-spark-full"), *tokens])
    args = parse_args(extra_args_provider=get_miles_extra_args_provider())
    args.rank, args.world_size, args.compress_ratios = 0, 8 * num_nodes, None
    args = set_default_megatron_args(args)
    miles_validate_args(args)
    validate_args(args)
    assert args.hf_checkpoint == "/root/models/Muse-Spark-Open-816B-A42B"
    assert args.ref_load == "/root/models/Muse-Spark-Open-816B-A42B_torch_dist"
    assert args.prompt_data == "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl"
    assert (args.rollout_batch_size, args.n_samples_per_prompt, args.global_batch_size) == (
        counts[1],
        counts[2],
        counts[1] * counts[2],
    )
    assert args.rollout_max_response_len == (4096 if optimizer == "muon" else 2048)
    if optimizer == "muon":
        assert args.apply_chat_template_kwargs["reasoning_effort"] == "xhigh"
    assert args.use_dynamic_batch_size and args.max_tokens_per_gpu == max_tokens
    assert args.use_rollout_routing_replay == use_r3
    assert args.use_routing_replay == args.use_rollout_routing_replay
    assert args.num_rollout == counts[0] and args.num_steps_per_rollout == 1
    assert args.dynamic_sampling_filter_path is None and not args.use_wandb
    tp, pp, cp, ep, edge_layers = layout
    assert (args.tensor_model_parallel_size, args.pipeline_model_parallel_size, args.expert_model_parallel_size) == (
        tp,
        pp,
        ep,
    )
    assert args.context_parallel_size == cp
    if cp > 1:
        from megatron.training.arguments import core_transformer_config_from_args

        from miles_plugins.models.muse_spark.provider import _validate_fused_context_parallel

        config = core_transformer_config_from_args(args)
        assert config.cp_comm_type == "a2a"
        _validate_fused_context_parallel(config)
    assert args.data_parallel_size == 1 and args.expert_tensor_parallel_size == 1
    assert args.num_layers == 62 and args.moe_layer_freq == [1] * 62
    assert args.decoder_first_pipeline_num_layers == args.decoder_last_pipeline_num_layers == edge_layers
    if optimizer == "adam":
        assert args.optimizer == "adam" and args.stream_optimizer_state_to_disk
        assert args.stream_optimizer_state_moment_dtype == "fp32" and args.offload_train_target == "disk"
    else:
        assert args.optimizer == "muon" and args.use_layer_wise_distributed_optimizer
        assert not args.stream_optimizer_state_to_disk
        assert args.offload_train_target == "cpu" and args.chunked_optimizer_state_offload
        assert args.chunked_muon_master_weights and args.optimizer_state_offload_fraction == 1
    assert args.loss_type == "policy_loss" and args.advantage_estimator == "grpo"
    assert args.calculate_per_token_loss and args.skip_actor_forward_only
    assert args.save is None and not args.debug_train_only and not args.debug_rollout_only
    assert (args.actor_num_nodes, args.actor_num_gpus_per_node) == (num_nodes, 8)
    assert (args.rollout_num_gpus_per_engine, args.sglang_ep_size, args.sglang_pp_size) == (
        gpus_per_engine,
        gpus_per_engine,
        1,
    )
    assert not args.sglang_disable_cuda_graph and args.sglang_mem_fraction_static == mem_fraction
    assert args.sglang_max_running_requests == running and args.sglang_cuda_graph_bs_decode == graph_bs
    assert args.save_debug_rollout_data.endswith("/rollout_data/{rollout_id}.pt")
    assert args.pin_rollout_manager_to_head == (script == _B300)


@pytest.mark.parametrize("script,layout", [(_H200, (8, 8, 1, 8, 7)), (_B300, (8, 4, 1, 8, 15))])
def test_entry_point_sets_its_cluster_and_engine_sizing(script, layout):
    args = import_launch_script(U.repo_base_dir / script).ScriptArgs()
    num_nodes, gpus_per_engine, max_tokens, mem_fraction, running, graph_bs = _HARDWARE[script]
    assert (args.num_nodes, args.num_gpus_per_node, args.rollout_num_gpus_per_engine) == (
        num_nodes,
        8,
        gpus_per_engine,
    )
    assert tuple(getattr(args, name) for name in _LAYOUT_FIELDS) == layout
    assert (args.max_tokens_per_gpu, args.sglang_mem_fraction_static, args.sglang_max_running_requests) == (
        max_tokens,
        mem_fraction,
        running,
    )
    assert list(args.sglang_cuda_graph_bs_decode) == graph_bs


def test_explicit_values_override_the_entry_point():
    args = import_launch_script(U.repo_base_dir / _B300).ScriptArgs(
        max_tokens_per_gpu=4096, sglang_mem_fraction_static=0.8, sglang_max_running_requests=64
    )
    assert (args.max_tokens_per_gpu, args.sglang_mem_fraction_static, args.sglang_max_running_requests) == (
        4096,
        0.8,
        64,
    )


def test_the_shared_recipe_takes_cluster_values_only_from_an_entry_point():
    common = import_launch_script(U.repo_base_dir / _B300).common
    with pytest.raises(TypeError, match="num_nodes"):
        common.ScriptArgs()


@pytest.mark.parametrize("script", [_H200, _B300])
@pytest.mark.parametrize("buffer_size", [4 * 1024**3, 8 * 1024**3])
def test_weight_buffer_override_reaches_train_entrypoint(monkeypatch, script, buffer_size):
    launcher = import_launch_script(U.repo_base_dir / script)
    tokens, _ = _train_args(monkeypatch, launcher, update_weight_buffer_size=buffer_size)
    assert tokens.count("--update-weight-buffer-size") == 1
    assert tokens[tokens.index("--update-weight-buffer-size") + 1] == str(buffer_size)


# --partial on each entry point: GPUs per node -> (TP, PP, CP, EP, layers on each edge stage).
@pytest.mark.parametrize("script,num_gpus,layout", [(_H200, 8, (4, 2, 1, 4, 2)), (_B300, 4, (2, 2, 1, 2, 2))])
def test_partial_runs_the_four_layer_checkpoint_on_one_node(monkeypatch, tmp_path, script, num_gpus, layout):
    from miles.backends.megatron_utils.arguments import parse_args, set_default_megatron_args, validate_args
    from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    launcher = import_launch_script(U.repo_base_dir / script)
    template = tmp_path / "chat.jinja"
    template.write_text("{{ messages }}")
    commands, conversions, calls = [], [], []
    monkeypatch.setattr(U, "exec_command_cpu", commands.append)
    monkeypatch.setattr(U, "convert_checkpoint", lambda **kw: conversions.append(kw))
    monkeypatch.setattr(U, "hf_download_dataset", lambda *a, **kw: None)
    monkeypatch.setattr(U, "execute_train", lambda **kw: calls.append(kw))
    launcher.main(partial=True, num_gpus_per_node=num_gpus, chat_template_path=str(template))

    assert commands == [
        "hf download RadixArk/Muse-Spark-Open-last4-partial --local-dir /root/models/Muse-Spark-Open-last4-partial"
    ]
    (conversion,) = conversions
    assert not conversion["multinode"] and conversion["megatron_model_type"] == "muse-spark-last4-partial"
    (call,) = calls
    assert call["config"].num_nodes == 1 and call["megatron_model_type"] == "muse-spark-last4-partial"

    blackwell = script == _B300
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3) if blackwell else (9, 0))
    monkeypatch.setattr("megatron.training.arguments.get_device_arch_version", lambda: 10 if blackwell else 9)
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    tokens = shlex.split(call["train_args"])
    monkeypatch.setattr(sys, "argv", ["train.py", *expand_model_args("muse-spark-last4-partial"), *tokens])
    args = parse_args(extra_args_provider=get_miles_extra_args_provider())
    args.rank, args.world_size, args.compress_ratios = 0, num_gpus, None
    args = set_default_megatron_args(args)
    miles_validate_args(args)
    validate_args(args)
    assert args.hf_checkpoint == "/root/models/Muse-Spark-Open-last4-partial"
    assert args.ref_load == "/root/models/Muse-Spark-Open-last4-partial_torch_dist"
    assert args.num_layers == 4
    tp, pp, cp, ep, edge_layers = layout
    assert (args.tensor_model_parallel_size, args.pipeline_model_parallel_size) == (tp, pp)
    assert (args.context_parallel_size, args.expert_model_parallel_size) == (cp, ep)
    assert args.decoder_first_pipeline_num_layers == args.decoder_last_pipeline_num_layers == edge_layers
    assert (args.actor_num_nodes, args.actor_num_gpus_per_node, args.rollout_num_gpus_per_engine) == (
        1,
        num_gpus,
        num_gpus,
    )
    assert (args.num_rollout, args.rollout_batch_size, args.n_samples_per_prompt) == (3, 4, 4)
    assert (args.rollout_max_response_len, args.rollout_max_context_len) == (1024, 4096)
    assert args.rm_type == "deterministic_random" and args.ci_test and not args.ci_disable_kl_checker
    assert args.check_weight_update_skip_list == [
        "model.vision_tower.",
        "model.vision_adapter.",
        "model.vision_projection.",
        "model.audio_tower.",
    ]


@pytest.mark.parametrize("script", [_H200, _B300])
def test_a_multi_node_entry_point_converts_across_the_ray_cluster(monkeypatch, script):
    conversions = []
    monkeypatch.setattr(U, "exec_command_cpu", lambda command: None)
    monkeypatch.setattr(U, "convert_checkpoint", lambda **kw: conversions.append(kw))
    monkeypatch.setattr(U, "hf_download_dataset", lambda *a, **kw: None)
    monkeypatch.setattr(U, "execute_train", lambda **kw: None)
    import_launch_script(U.repo_base_dir / script).main()
    (conversion,) = conversions
    assert conversion["multinode"] and conversion["megatron_model_type"] == "muse-spark-full"
