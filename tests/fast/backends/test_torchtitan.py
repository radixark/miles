import importlib.util
import json
import re
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles import platforms
from miles.backends.torchtitan_utils import arguments
from miles.backends.torchtitan_utils import parallel as tp
from miles.backends.torchtitan_utils.args_namespace import TorchtitanArgsNamespace
from miles.backends.torchtitan_utils.arguments import validate_torchtitan_args
from miles.utils.arguments import parse_args_and_get_parser
from miles.utils.hf_utils.weight_mapping import HfWeightMapping
from miles.utils.lora import arguments as lora_arguments

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])


def _args(**overrides) -> Namespace:
    base = dict(
        titan_model_name="qwen3",
        seq_length=8192,
        micro_batch_size=1,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        log_probs_backend="torch",
        colocate=False,
        update_weight_transfer_mode="broadcast",
        titan_fsdp_gather_once=False,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        max_seq_len=8192,
        rollout_max_response_len=4096,
        ref_update_interval=None,
        save_debug_train_data=None,
        ci_test=False,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
        optimizer_cpu_offload=False,
        stream_optimizer_state_to_disk=False,
        stream_optimizer_state_moment_dtype="fp32",
        fp16=False,
        lr_decay_style="constant",
        lr_warmup_fraction=None,
        lr_wsd_decay_iters=None,
        lr_decay_iters=None,
        lora_rank=0,
        lora_type="lora",
        lora_A_init_method="kaiming",
        lora_dropout=0.0,
        lora_adapter_path=None,
        lora_train_only=False,
        colocate_memory_peak_device="cpu",
        titan_fsdp_reduce_dtype="float32",
        # read by the padded-rows loss layout's startup check on a static-shape platform
        train_backend="torchtitan",
        loss_type="policy_loss",
        qkv_format="thd",
        advantage_estimator="grpo",
        entropy_coef=0.0,
        tensor_model_parallel_size=1,
        true_on_policy_mode=False,
        allgather_cp=False,
        use_rollout_sampling_support_replay=False,
        use_opsm=False,
        use_tis=False,
        get_mismatch_metrics=False,
        use_opd=False,
        custom_pg_loss_reducer_function_path=None,
        dump_details=None,
        multi_lora=False,
    )
    return Namespace(**{**base, **overrides})


def test_the_backend_needs_torch_214(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="torch>=2.14"):
        validate_torchtitan_args(_args())
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    validate_torchtitan_args(_args())


def test_ici_weight_sync_needs_the_shared_slice_env(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    monkeypatch.delenv("MILES_TPU_SHARED_SLICE", raising=False)
    with pytest.raises(ValueError, match="MILES_TPU_SHARED_SLICE"):
        validate_torchtitan_args(_args(update_weight_transfer_mode="ici"))
    monkeypatch.setenv("MILES_TPU_SHARED_SLICE", "0,8,localhost:29600")
    validate_torchtitan_args(_args(update_weight_transfer_mode="ici"))


def test_the_sequence_must_cover_the_rotary_tables(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    with pytest.raises(ValueError, match="seq-length"):
        validate_torchtitan_args(_args(seq_length=8192, max_seq_len=16384))
    with pytest.raises(ValueError, match="no room for a prompt"):
        validate_torchtitan_args(_args(seq_length=8192, max_seq_len=None, rollout_max_response_len=8192))
    validate_torchtitan_args(_args(seq_length=16384, max_seq_len=None, rollout_max_response_len=8192))


_TPU_FACTS = (
    "dynamic_shapes",
    "compile_backend",
    "triton_kernels",
    "flex_attention",
    "rooted_reduce",
    "premul_sum_reduce",
    "weight_transfer_modes",
)


@pytest.fixture
def on_tpu(monkeypatch):
    """The process platform takes TpuPlatform's facts, read from its module loaded against a stub torch_tpu."""
    monkeypatch.setitem(sys.modules, "torch_tpu", ModuleType("torch_tpu"))
    spec = importlib.util.spec_from_file_location("_test_tpu_platform", Path(platforms.__file__).with_name("tpu.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for fact in _TPU_FACTS:
        monkeypatch.setattr(platforms.current_platform, fact, getattr(module.TpuPlatform, fact))
    # TPU trainers sync weights over ICI, which needs the slice they share with the engine
    monkeypatch.setenv("MILES_TPU_SHARED_SLICE", "0,8,localhost:29600")


def _tpu_args(**overrides) -> Namespace:
    return _args(update_weight_transfer_mode="ici", **overrides)


def test_a_tpu_microbatch_cannot_outgrow_the_largest_length_bucket(monkeypatch, on_tpu):
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    with pytest.raises(ValueError, match="max-tokens-per-gpu 16384 exceeds"):
        validate_torchtitan_args(_tpu_args(use_dynamic_batch_size=True, max_tokens_per_gpu=16384))
    validate_torchtitan_args(_tpu_args(use_dynamic_batch_size=True, max_tokens_per_gpu=8192))
    # fixed-size microbatches pack micro_batch_size samples, each up to --seq-length
    with pytest.raises(ValueError, match="micro-batch-size 2"):
        validate_torchtitan_args(_tpu_args(micro_batch_size=2))
    validate_torchtitan_args(_tpu_args(micro_batch_size=1))


def test_tpu_sequence_length_is_a_whole_number_of_attention_blocks(monkeypatch, on_tpu):
    """The largest length bucket is --seq-length, and the TPU attention kernel reshapes it into 1024-token blocks."""
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    with pytest.raises(ValueError, match="seq-length 3000 must be a multiple of 1024"):
        validate_torchtitan_args(_tpu_args(seq_length=3000, max_seq_len=None, rollout_max_response_len=2048))
    validate_torchtitan_args(_tpu_args(seq_length=8192))


def test_tpu_refuses_context_and_expert_parallelism(monkeypatch, on_tpu):
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    with pytest.raises(ValueError, match="context-parallel-size 2"):
        validate_torchtitan_args(_tpu_args(context_parallel_size=2))
    with pytest.raises(ValueError, match="expert-model-parallel-size 2"):
        validate_torchtitan_args(_tpu_args(expert_model_parallel_size=2))


def test_the_triton_log_prob_kernel_is_refused_on_tpu(monkeypatch, on_tpu):
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    with pytest.raises(ValueError, match="Triton"):
        validate_torchtitan_args(_tpu_args(log_probs_backend="fused"))


def test_tpu_weights_reach_disaggregated_engines_over_ici_only(monkeypatch, on_tpu):
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    for unsupported in (dict(update_weight_transfer_mode="broadcast"), dict(colocate=True)):
        with pytest.raises(ValueError, match="update-weight-transfer-mode ici"):
            validate_torchtitan_args(_args(**{"update_weight_transfer_mode": "ici", **unsupported}))
    validate_torchtitan_args(_tpu_args())


def test_gpu_validation_ignores_the_tpu_limits(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    validate_torchtitan_args(
        _args(use_dynamic_batch_size=True, max_tokens_per_gpu=16384, log_probs_backend="fused", colocate=True)
    )


@pytest.mark.parametrize("seq_len", [1000, 1024, 3000, 8192])
def test_the_recompile_limit_counts_every_length_bucket(seq_len):
    """A bucket the limit misses is one compile too many, which a fullgraph block turns into a crash."""
    from miles.utils.data import fixed_shape_length_bucket, fixed_shape_length_buckets

    buckets = {fixed_shape_length_bucket(n, seq_len) for n in range(1, seq_len + 1)}
    assert buckets == fixed_shape_length_buckets(seq_len)
    assert max(buckets) == seq_len


@pytest.mark.parametrize(("n_tokens", "padded"), [(1, 1024), (1024, 1024), (1025, 2048), (5000, 8192), (8192, 8192)])
def test_a_static_shape_platform_pads_each_microbatch_to_a_length_bucket(monkeypatch, n_tokens, padded):
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.trainer import TitanTrainer

    trainer = SimpleNamespace(
        parallelism=SimpleNamespace(pp_enabled=False, cp_enabled=False),
        config=SimpleNamespace(training=SimpleNamespace(max_context_length=8192)),
    )
    assert TitanTrainer.padded_length(trainer, n_tokens) == n_tokens, "a dynamic-shape platform keeps the length"
    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", False)
    assert TitanTrainer.padded_length(trainer, n_tokens) == padded
    with pytest.raises(ValueError, match="largest length bucket"):
        TitanTrainer.padded_length(trainer, 8193)


def test_compiled_blocks_share_one_graph_across_layers():
    """Each block's forward is compiled in place; a graph per layer would multiply the recompile budget."""
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.trainer import _compile_blocks

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, x):
            return torch.relu(self.linear(x)) + x

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleDict({str(i): Block() for i in range(3)})
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    _compile_blocks([model], backend="eager")
    x = torch.randn(4, 8)
    for block in model.layers.values():
        torch.testing.assert_close(block(x), Block.forward(block, x))
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1


def test_a_static_shape_platform_compiles_static_graphs_for_every_pass_and_bucket(monkeypatch):
    """Each pass with its own guards compiles once per bucket, and none may turn dynamic, which TPU rejects."""
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.trainer import TitanTrainer

    trainer = SimpleNamespace(config=SimpleNamespace(training=SimpleNamespace(max_context_length=8192)))

    def configure_fixed_shape_compile(*, num_passes):
        TitanTrainer.configure_fixed_shape_compile(trainer, num_passes=num_passes)

    # dynamo's own patch: monkeypatch's undo deletes a config key, which hides it from every later compile
    with torch._dynamo.config.patch(recompile_limit=1, automatic_dynamic_shapes=True):
        configure_fixed_shape_compile(num_passes=3)
        assert torch._dynamo.config.recompile_limit == 1, "a dynamic-shape platform compiles one graph per pass"
        assert torch._dynamo.config.automatic_dynamic_shapes

        monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", False)
        configure_fixed_shape_compile(num_passes=2)
        # buckets 1024, 2048, 4096, 8192, and two block variants (dense and MoE) per class
        assert torch._dynamo.config.recompile_limit == 2 * 2 * 4
        assert not torch._dynamo.config.automatic_dynamic_shapes
        configure_fixed_shape_compile(num_passes=3)
        assert torch._dynamo.config.recompile_limit == 2 * 3 * 4


def test_context_parallelism_is_rejected_for_qwen3_5(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    with pytest.raises(ValueError, match="context parallelism"):
        validate_torchtitan_args(_args(titan_model_name="qwen3_5", context_parallel_size=2))


def test_unsupported_flags_are_rejected_rather_than_ignored(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    with pytest.raises(ValueError, match="ref-update-interval"):
        validate_torchtitan_args(_args(ref_update_interval=4))
    with pytest.raises(ValueError, match="save-debug-train-data"):
        validate_torchtitan_args(_args(save_debug_train_data="/tmp/dump"))


def test_the_replay_checker_is_refused_where_blocks_compile_fullgraph(monkeypatch):
    """The checker syncs to the host from inside a block; a static-shape platform compiles blocks fullgraph."""
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    monkeypatch.setattr(arguments.current_platform, "dynamic_shapes", False)
    rollout_replay = dict(use_routing_replay=True, use_rollout_routing_replay=True)
    validate_torchtitan_args(_args(**rollout_replay))
    with pytest.raises(ValueError, match="ci-test"):
        validate_torchtitan_args(_args(ci_test=True, **rollout_replay))
    monkeypatch.setattr(arguments.current_platform, "dynamic_shapes", True)
    validate_torchtitan_args(_args(ci_test=True, **rollout_replay))


def test_recording_routes_in_a_compiled_block_is_refused_without_dynamic_shapes(monkeypatch):
    """Recording appends to the queue inside the block, so every microbatch recompiles on a static-shape platform."""
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    monkeypatch.setattr(arguments.current_platform, "dynamic_shapes", False)
    with pytest.raises(ValueError, match="use-rollout-routing-replay"):
        validate_torchtitan_args(_args(use_routing_replay=True))
    monkeypatch.setattr(arguments.current_platform, "dynamic_shapes", True)
    validate_torchtitan_args(_args(use_routing_replay=True))


def test_lora_flags_torchtitan_cannot_honor_are_rejected(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    validate_torchtitan_args(_args(lora_rank=8))
    with pytest.raises(ValueError, match="qwen3 only"):
        validate_torchtitan_args(_args(lora_rank=8, titan_model_name="glm5_next"))
    with pytest.raises(ValueError, match="--lora-type lora"):
        validate_torchtitan_args(_args(lora_rank=8, lora_type="canonical_lora"))
    with pytest.raises(ValueError, match="kaiming"):
        validate_torchtitan_args(_args(lora_rank=8, lora_A_init_method="xavier"))
    with pytest.raises(ValueError, match="no dropout"):
        validate_torchtitan_args(_args(lora_rank=8, lora_dropout=0.05))
    with pytest.raises(ValueError, match="grad buffers"):
        validate_torchtitan_args(_args(lora_rank=8, colocate_memory_peak_device="gpu"))
    validate_torchtitan_args(_args(lora_rank=8, colocate_memory_peak_device="gpu", lora_train_only=True))


def test_lora_defaults_are_the_values_torchtitan_honors(tmp_path, monkeypatch):
    """A torchtitan LoRA command without --lora-type or --lora-A-init-method must not hit the refusals.

    Parsed end to end: validate_torchtitan_args refuses any --lora-type but lora, so it only passes after
    miles_validate_args has resolved the default. torchtitan runs in the default raw mode, so a default
    keyed on raw mode alone would pick canonical_lora.
    """
    from transformers import Qwen3Config

    Qwen3Config(num_hidden_layers=1).save_pretrained(tmp_path)
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    monkeypatch.setattr(
        lora_arguments.HfWeightMapping,
        "from_config",
        lambda config: HfWeightMapping(frozenset({"model.layers.0.self_attn.q_proj.weight"})),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test",
            "--train-backend",
            "torchtitan",
            "--hf-checkpoint",
            str(tmp_path),
            "--lora-rank",
            "8",
            "--target-modules",
            "q_proj",
            "--rollout-batch-size",
            "2",
            "--num-rollout",
            "1",
            "--actor-num-gpus-per-node",
            "1",
            "--micro-batch-size",
            "1",
            "--rollout-max-response-len",
            "1024",
        ],
    )

    args, _ = parse_args_and_get_parser()

    assert (args.lora_type, args.lora_A_init_method) == ("lora", "kaiming")


def test_lora_targets_bypass_megatrons_native_spec(monkeypatch):
    """torchtitan runs in the default raw mode; megatron's native adapter spec must not vet its targets."""
    hf_config = SimpleNamespace(model_type="qwen3", to_dict=lambda: {"model_type": "qwen3"})
    monkeypatch.setattr(lora_arguments, "load_hf_config", lambda path: hf_config)
    monkeypatch.setattr(
        lora_arguments.HfWeightMapping,
        "from_config",
        lambda config: HfWeightMapping(frozenset({"model.layers.0.self_attn.q_proj.weight"})),
    )
    monkeypatch.setattr(
        lora_arguments,
        "resolve_adapter_targets",
        lambda *args, **kwargs: pytest.fail("torchtitan LoRA targets went through megatron's native spec"),
    )
    args = Namespace(
        train_backend="torchtitan",
        megatron_to_hf_mode="raw",
        hf_checkpoint="/hf",
        lora_rank=8,
        lora_alpha=None,
        lora_adapter_path=None,
        lora_adapter_convert_save=None,
        lora_type="lora",
        target_modules="q_proj",
        exclude_modules=None,
        experts_shared_outer_loras=False,
        multi_lora_n_adapters=0,
    )
    lora_arguments.validate_lora_args(args)
    assert args.lora_adapter_targets == ["q_proj"]


def _config_args(**overrides: object) -> Namespace:
    base = dict(
        optimizer="adam",
        titan_model_name="qwen3",
        titan_model_flavor="0.6B",
        seq_length=4096,
        dp_replicate_size=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        global_batch_size=8,
        micro_batch_size=1,
        clip_grad=1.0,
        lr=1e-6,
        min_lr=0.0,
        lr_warmup_iters=0,
        lr_decay_style="constant",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_eps=1e-8,
        weight_decay=0.1,
        seed=1,
        gradient_checkpointing=False,
        optimizer_cpu_offload=False,
        titan_fsdp_gather_once=False,
        titan_fsdp_reduce_dtype="float32",
        save=None,
        load=None,
    )
    return Namespace(backend=TorchtitanArgsNamespace(**{**base, **overrides}))


def _checkpoint_dir(tmp_path, **config) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3", **config}))
    return str(tmp_path)


@pytest.fixture
def single_gpu_dims(monkeypatch):
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils import config as titan_config

    monkeypatch.setattr(titan_config.dist, "get_world_size", lambda group=None: 1)


def test_a_tied_checkpoint_is_refused_under_pipeline_parallelism(tmp_path, single_gpu_dims):
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    hf = _checkpoint_dir(tmp_path, tie_word_embeddings=True)
    with pytest.raises(ValueError, match="pipeline"):
        build_trainer_config(
            _config_args(pipeline_model_parallel_size=2), hf_assets_path=hf, lr_total_steps=1, dump_subdir="x"
        )


def test_the_lr_schedule_follows_miles_flags_not_torchtitan_defaults(tmp_path, single_gpu_dims):
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    hf = _checkpoint_dir(tmp_path, tie_word_embeddings=False)
    constant = build_trainer_config(
        _config_args(lr_warmup_iters=3), hf_assets_path=hf, lr_total_steps=10, dump_subdir="x"
    )
    assert (constant.optim.lr_scheduler.warmup_steps, constant.optim.lr_scheduler.min_lr_factor) == (3, 1.0)
    cosine = build_trainer_config(
        _config_args(lr_decay_style="cosine", lr=1e-6, min_lr=1e-7),
        hf_assets_path=hf,
        lr_total_steps=10,
        dump_subdir="x",
    )
    assert cosine.optim.lr_scheduler.decay_type == "cosine"
    assert cosine.optim.lr_scheduler.min_lr_factor == pytest.approx(0.1)


def _layer_linears_accumulate_in_place(model) -> bool:
    from torchtitan.models.common.linear import Linear

    return any(
        linear.inplace_wgrad_accum for layer in model.layers for _, linear, _, _ in layer.traverse(Linear.Config)
    )


def test_a_static_shape_platform_always_compiles_its_blocks(tmp_path, single_gpu_dims, on_tpu):
    """Eager torch_tpu runs every op as its own program, so TPU compiles its blocks with no flag asking for it."""
    from miles.backends.torchtitan_utils.config import build_trainer_config

    config = build_trainer_config(
        _config_args(),
        hf_assets_path=_checkpoint_dir(tmp_path, tie_word_embeddings=False),
        lr_total_steps=1,
        dump_subdir="x",
    )
    assert config.compile_backend == "tpu"
    assert config.model.local_compile_regions == []
    assert not _layer_linears_accumulate_in_place(config.model), "a traced backward cannot add into .grad in place"


def test_gpu_runs_its_blocks_eager(tmp_path, single_gpu_dims):
    from miles.backends.torchtitan_utils.config import build_trainer_config

    hf = _checkpoint_dir(tmp_path, tie_word_embeddings=False)
    eager = build_trainer_config(_config_args(), hf_assets_path=hf, lr_total_steps=1, dump_subdir="x")
    assert eager.compile_backend is None and eager.model.local_compile_regions
    assert _layer_linears_accumulate_in_place(eager.model)


def test_a_platform_without_sparse_flex_attention_gets_tpu_flash_attention_in_every_layer(
    tmp_path, single_gpu_dims, on_tpu
):
    from miles.backends.torchtitan_utils.config import TpuFlashInnerAttention, build_trainer_config

    config = build_trainer_config(
        _config_args(),
        hf_assets_path=_checkpoint_dir(tmp_path, tie_word_embeddings=False),
        lr_total_steps=1,
        dump_subdir="x",
    )
    inner = {type(layer.attention.inner_attention) for layer in config.model.layers}
    assert inner == {TpuFlashInnerAttention.Config}


def test_tpu_attention_segments_restart_where_the_positions_do():
    """The kernel masks across segments, so a segment must start exactly where a packed sequence (or the pad) does."""
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import TpuFlashInnerAttention

    positions = torch.cat([torch.arange(n) for n in (700, 900, 1200, 1296)])
    segment_ids = TpuFlashInnerAttention.Config().build_attention_metadata(positions)
    assert segment_ids.dtype == torch.int32
    assert torch.all(segment_ids[1:] >= segment_ids[:-1]), "the kernel skips blocks assuming non-decreasing ids"
    starts = (positions == 0).cumsum(-1)
    assert torch.equal(segment_ids[1:] != segment_ids[:-1], starts[1:] != starts[:-1])


def test_mla_keeps_its_wrapper_and_runs_tpu_flash_attention_inside_it(on_tpu):
    """MLA wraps the flex kernel in its own inner attention; only the kernel has a TPU replacement."""
    pytest.importorskip("torchtitan")
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.common.attention import MLAAttention

    from miles.backends.torchtitan_utils.config import TpuFlashInnerAttention, build_model_config

    model = build_model_config(
        _config_args(titan_model_name="glm4_moe_lite", titan_model_flavor="4layer"), parallelism=ParallelismConfig()
    )
    assert len(model.layers) == 4
    mla = [layer.attention.inner_attention for layer in model.layers]
    assert {type(wrapper) for wrapper in mla} == {MLAAttention.Config}
    assert {type(wrapper.inner_attention) for wrapper in mla} == {TpuFlashInnerAttention.Config}


def test_context_parallelism_swaps_in_the_all_gather_kv_attention(tmp_path, monkeypatch):
    """torchtitan refuses CP on a plain flex attention; the backend converts every layer's inner attention."""
    pytest.importorskip("torchtitan")
    from torchtitan.models.common.attention.cp_attention import AllGatherKVFlexInnerAttention

    from miles.backends.torchtitan_utils import config as titan_config

    monkeypatch.setattr(titan_config.dist, "get_world_size", lambda group=None: 2)
    config = titan_config.build_trainer_config(
        _config_args(context_parallel_size=2),
        hf_assets_path=_checkpoint_dir(tmp_path, tie_word_embeddings=False),
        lr_total_steps=10,
        dump_subdir="x",
    )
    assert config.parallelism.context_parallel_degree == 2
    assert config.parallelism.context_parallel_load_balancer is None
    inner = {type(layer.attention.inner_attention) for layer in config.model.layers}
    assert inner == {AllGatherKVFlexInnerAttention.Config}


class _Mesh:
    def __init__(self, size: int):
        self._group = f"pg(size={size})"

    def get_group(self):
        return self._group


class _ParallelDims:
    def __init__(self, meshes: dict[str, int]):
        self._meshes = {name: _Mesh(size) for name, size in meshes.items()}

    def get_optional_mesh(self, name):
        return self._meshes.get(name)

    def get_mesh(self, name):
        return self._meshes[name]


@pytest.fixture
def dist_stub():
    sizes: dict = {}

    def get_world_size(group=None):
        return sizes.get(group, sizes.get("__world__", 1))

    with (
        patch.object(tp.dist, "get_rank", lambda group=None: 0),
        patch.object(tp.dist, "get_world_size", get_world_size),
        patch.object(
            tp.dist,
            "new_group",
            lambda ranks, backend=None: "self_group" if backend is None else f"gloo_sub{tuple(ranks)}",
        ),
        patch.object(tp.dist, "new_subgroups_by_enumeration", lambda ranks_per_subgroup_list: ("self_group", [])),
        patch.object(tp.dist, "get_process_group_ranks", lambda group: list(range(sizes.get(group, 1)))),
        patch.object(
            tp.dist, "all_gather_object", lambda out, obj, group=None: out.__setitem__(slice(None), [obj] * len(out))
        ),
        patch.object(tp, "get_gloo_group", lambda: "gloo"),
    ):
        yield sizes


def _state(dist_stub, meshes: dict[str, int], world: int | None = None, **kwargs):
    for size in meshes.values():
        dist_stub[f"pg(size={size})"] = size
    world = world if world is not None else meshes.get("loss", 1)
    dist_stub["__world__"] = world
    dist_stub["gloo"] = world
    dist_stub["self_group"] = 1
    dp = meshes.get("dp", 1)
    dist_stub[f"gloo_sub{tuple(range(dp))}"] = dp
    return tp.create_titan_parallel_state(_ParallelDims(meshes), **kwargs)


def test_context_parallelism_stays_inside_the_trainer_and_absent_axes_are_trivial(dist_stub):
    state = _state(dist_stub, {"dp": 4, "loss": 8, "cp": 2})
    assert (state.intra_dp.size, state.intra_dp_cp.size, state.cp.size) == (4, 4, 1)
    for axis in (state.tp, state.pp, state.ep, state.etp, state.indep_dp):
        assert (axis.size, axis.rank) == (1, 0)


def test_every_absent_axis_shares_one_collectively_built_singleton_group(dist_stub):
    """tpu_dist needs every rank to create every subgroup in the same order; a per-rank new_group breaks that."""
    with patch.object(tp.dist, "new_group", side_effect=AssertionError("a singleton group from new_group")):
        state = _state(dist_stub, {"dp": 1, "loss": 1})
    absent_axes = (state.cp, state.tp, state.pp, state.ep, state.etp, state.indep_dp)
    assert {axis.group for axis in absent_axes} == {"self_group"}


def test_the_loss_adapter_undoes_the_summed_dp_and_cp_gradients():
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.loss import RLLossAdapter

    adapter = RLLossAdapter(RLLossAdapter.Config())
    adapter.set_gradient_scale(1.0 / 8)
    batch = {"tokens": torch.zeros(1)}
    adapter.arm([batch], lambda pred, b: (pred.sum(), {"seen": b is batch}), is_training=True)
    loss, _ = adapter(torch.full((2,), 4.0), torch.zeros(2, dtype=torch.long))
    assert loss.item() == pytest.approx(1.0)
    assert adapter.collect() == [{"seen": True}]


def test_resume_reads_from_load_and_writes_to_save(tmp_path, single_gpu_dims):
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    hf = _checkpoint_dir(tmp_path / "hf", tie_word_embeddings=False)
    load_root = tmp_path / "load"
    for step in (3, 12):
        (load_root / "torchtitan" / "actor" / "checkpoint" / f"step-{step}").mkdir(parents=True)
    save_root = tmp_path / "save"

    config = build_trainer_config(
        _config_args(load=str(load_root), save=str(save_root)),
        hf_assets_path=hf,
        lr_total_steps=1,
        dump_subdir="actor",
    )
    assert config.output_dir == str(save_root / "torchtitan" / "actor")
    assert config.checkpointer.initial_load_path == str(load_root / "torchtitan" / "actor" / "checkpoint" / "step-12")
    assert (config.checkpointer.initial_load_model_only, config.checkpointer.initial_load_in_hf) == (False, False)
    assert config.checkpointer.last_save_model_only is False

    fresh = build_trainer_config(
        _config_args(load=str(tmp_path / "empty"), save=str(save_root)),
        hf_assets_path=hf,
        lr_total_steps=1,
        dump_subdir="actor",
    )
    assert fresh.checkpointer.initial_load_path is None
    assert (fresh.checkpointer.initial_load_model_only, fresh.checkpointer.initial_load_in_hf) == (True, True)


def test_routing_replay_hooks_the_router_where_it_picks_experts():
    """routing_replay swaps TokenChoiceTopKRouter._select_experts per router; that only replays the routing if the
    upstream forward still looks the method up on the instance and gathers its scores from the returned ids."""
    import inspect

    moe = pytest.importorskip("torchtitan.models.common.moe")

    forward = inspect.getsource(moe.TokenChoiceTopKRouter.forward)
    assert "self._select_experts," in forward
    assert "scores_TE.gather(dim=-1, index=topk_expert_ids_TK)" in forward
    assert list(inspect.signature(moe.TokenChoiceTopKRouter._select_experts).parameters) == [
        "self",
        "scores_TE",
        "expert_bias_E",
        "router_kwargs",
    ]


_GLM47_FLASH_HF_CONFIG = dict(
    hidden_size=2048,
    num_hidden_layers=47,
    num_attention_heads=20,
    q_lora_rank=768,
    kv_lora_rank=512,
    qk_nope_head_dim=192,
    qk_rope_head_dim=64,
    v_head_dim=256,
    intermediate_size=10240,
    moe_intermediate_size=1536,
    n_routed_experts=64,
    n_shared_experts=1,
    num_experts_per_tok=4,
    first_k_dense_replace=1,
    routed_scaling_factor=1.8,
    norm_topk_prob=True,
    rope_theta=1000000,
    rms_norm_eps=1e-5,
    vocab_size=154880,
)

_GLM47_FLASH_CHECKPOINT_KEYS = [
    "lm_head.weight",
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.layers.{}.input_layernorm.weight",
    "model.layers.{}.post_attention_layernorm.weight",
    "model.layers.{}.self_attn.q_a_proj.weight",
    "model.layers.{}.self_attn.q_a_layernorm.weight",
    "model.layers.{}.self_attn.q_b_proj.weight",
    "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight",
    "model.layers.{}.self_attn.kv_a_layernorm.weight",
    "model.layers.{}.self_attn.kv_b_proj.weight",
    "model.layers.{}.self_attn.o_proj.weight",
    "model.layers.{}.mlp.gate_proj.weight",
    "model.layers.{}.mlp.up_proj.weight",
    "model.layers.{}.mlp.down_proj.weight",
    "model.layers.{}.mlp.gate.weight",
    "model.layers.{}.mlp.gate.e_score_correction_bias",
    "model.layers.{}.mlp.experts.{}.gate_proj.weight",
    "model.layers.{}.mlp.experts.{}.up_proj.weight",
    "model.layers.{}.mlp.experts.{}.down_proj.weight",
    "model.layers.{}.mlp.shared_experts.gate_proj.weight",
    "model.layers.{}.mlp.shared_experts.up_proj.weight",
    "model.layers.{}.mlp.shared_experts.down_proj.weight",
]


def test_unknown_model_names_name_both_package_roots():
    pytest.importorskip("torchtitan")
    from torchtitan.config.parallelism import ParallelismConfig

    from miles.backends.torchtitan_utils.config import build_model_config

    with pytest.raises(ValueError, match="miles.backends.torchtitan_utils.models.nope.*torchtitan.models.nope"):
        build_model_config(
            _config_args(titan_model_name="nope", titan_model_flavor="x"), parallelism=ParallelismConfig()
        )


def test_glm4_7_flash_flavor_matches_its_hf_config_and_its_checkpoint_keys():
    pytest.importorskip("torchtitan")
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.common import RMSNorm, Sigmoid
    from torchtitan.models.common.moe import MoE
    from torchtitan.models.deepseek_v3 import DeepSeekV3Model

    from miles.backends.torchtitan_utils.config import build_model_config

    model = build_model_config(
        _config_args(titan_model_name="glm4_moe_lite", titan_model_flavor="30B-A3B"), parallelism=ParallelismConfig()
    )
    hf = _GLM47_FLASH_HF_CONFIG
    assert (model.dim, model.vocab_size, len(model.layers)) == (
        hf["hidden_size"],
        hf["vocab_size"],
        hf["num_hidden_layers"],
    )
    attention = model.layers[0].attention
    assert attention.n_heads == hf["num_attention_heads"]
    assert (
        attention.q_lora_rank,
        attention.kv_lora_rank,
        attention.qk_nope_head_dim,
        attention.qk_rope_head_dim,
        attention.v_head_dim,
    ) == (hf["q_lora_rank"], hf["kv_lora_rank"], hf["qk_nope_head_dim"], hf["qk_rope_head_dim"], hf["v_head_dim"])
    assert attention.rope.theta == hf["rope_theta"] and attention.rope.scaling == "none"
    assert {norm.eps for _, norm, _, _ in model.traverse(RMSNorm.Config, recurse=True)} == {hf["rms_norm_eps"]}
    assert [isinstance(layer.feed_forward, MoE.Config) for layer in model.layers] == [
        i >= hf["first_k_dense_replace"] for i in range(47)
    ]
    assert model.layers[0].feed_forward.w2.in_features == hf["intermediate_size"]
    moe = model.layers[1].feed_forward
    assert moe.num_experts == hf["n_routed_experts"]
    assert moe.routed_experts.w2.in_features == hf["moe_intermediate_size"]
    assert moe.shared_experts.w2.in_features == hf["moe_intermediate_size"] * hf["n_shared_experts"]
    assert (moe.router.top_k, type(moe.router.score_func), moe.router.route_scale, moe.router.route_norm) == (
        hf["num_experts_per_tok"],
        Sigmoid.Config,
        hf["routed_scaling_factor"],
        hf["norm_topk_prob"],
    )
    assert (moe.router.aux_loss, moe.freeze_expert_bias) == (None, True)
    assert model.mtp_layers == []

    adapter = DeepSeekV3Model.state_dict_adapter_cls(model, None)
    assert set(_GLM47_FLASH_CHECKPOINT_KEYS) <= set(adapter.from_hf_map)


@pytest.mark.parametrize("gather_pp", [True, False])
def test_a_pipeline_stage_without_mla_layers_streams_past_the_q_lora_group(gather_pp):
    """A stage that streams only its own layers may hold none of a group's members (an all-KDA GLM-5.3 stage)."""
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.hf_weight_iterator import TitanHfWeightIterator
    from miles.backends.training_utils.weight_update.hf_weight_iterator.bucketing import assemble_atomic_update_groups

    iterator = SimpleNamespace(
        placement=SimpleNamespace(gather_pp=gather_pp), model_name="glm5_next", _q_lora_rank=1536
    )
    groups = TitanHfWeightIterator._hf_atomic_update_groups(iterator)
    kda_only_stage = [[("model.language_model.layers.0.self_attn.q_proj.weight", torch.zeros(1))]]

    if gather_pp:
        with pytest.raises(AssertionError, match="matched no params"):
            list(assemble_atomic_update_groups(kda_only_stage, groups))
    else:
        assert list(assemble_atomic_update_groups(kda_only_stage, groups)) == kda_only_stage


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_sleep_and_wake_leave_cpu_offloaded_training_state_on_the_host(monkeypatch, cpu_offload):
    """Under --optimizer-cpu-offload the shards and moments already live on the host; moving them on wake would pull
    every one onto the GPU."""
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils import actor as titan_actor
    from miles.backends.training_utils.torch_native import actor as native_actor

    moved: list[bool] = []
    monkeypatch.setattr(native_actor, "move_train_state", lambda *_args, to_host: moved.append(to_host))
    monkeypatch.setattr(titan_actor.dist, "barrier", lambda **_kwargs: None)
    for module in (titan_actor, native_actor):
        monkeypatch.setattr(module, "clear_memory", lambda: None)
        monkeypatch.setattr(module, "print_memory", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(module, "get_gloo_group", lambda: None)
    actor = titan_actor.TorchtitanTrainRayActor.__new__(titan_actor.TorchtitanTrainRayActor)
    actor.args = SimpleNamespace(offload_train=True, backend=SimpleNamespace(optimizer_cpu_offload=cpu_offload))
    actor.model_parts, actor.optimizers = [], []

    actor.sleep()
    actor.wake_up()

    assert moved == ([] if cpu_offload else [True, False])


class _SplittingExpertAdapter:
    """Converts grouped experts as DeepSeek-V3's adapter does: w13 is split into w1_EFD and w3_EFD before it records
    each one's local expert range."""

    def __init__(self, start: int, end: int) -> None:
        self.local_experts_indices: dict[str, tuple[int, int]] = {}
        self._span = (start, end)

    def to_hf(self, state):
        (key, value), *_ = state.items()
        template = re.sub(r"(\d+)", "{}", key, count=1)
        hf = {}
        for name in ("w1_EFD", "w3_EFD"):
            self.local_experts_indices[template.replace("w13.weight", name)] = self._span
            hf[f"model.layers.1.mlp.experts.{self._span[0]}.{name}"] = value[0]
        return hf


def test_local_experts_find_their_range_under_the_names_the_adapter_converted_to(monkeypatch):
    """GLM-4.7-Flash's ICI export died on a KeyError: its adapter records w13's range as w1_EFD's and w3_EFD's."""
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils import hf_weight_iterator

    monkeypatch.setattr(hf_weight_iterator.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(hf_weight_iterator.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(
        hf_weight_iterator, "current_platform", SimpleNamespace(current_device=lambda: torch.device("cpu"))
    )
    key = "layers.1.feed_forward.routed_experts.w13.weight"
    state = {key: torch.zeros(16, 2, 3)}

    exported = dict(hf_weight_iterator._local_experts(_SplittingExpertAdapter(4, 8), state, [key]))
    assert set(exported) == {"model.layers.1.mlp.experts.4.w1_EFD", "model.layers.1.mlp.experts.4.w3_EFD"}

    with pytest.raises(AssertionError, match="not the engine's even split"):
        list(hf_weight_iterator._local_experts(_SplittingExpertAdapter(0, 4), state, [key]))


def test_fsdp_gather_once_is_refused_under_pipeline_parallelism(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    with pytest.raises(ValueError, match="gather-once"):
        validate_torchtitan_args(_args(titan_fsdp_gather_once=True, pipeline_model_parallel_size=2))
    validate_torchtitan_args(_args(titan_fsdp_gather_once=True))


@pytest.mark.parametrize("gather_once", [False, True])
def test_gather_once_reduces_gradients_on_the_last_microbatch_only(monkeypatch, gather_once):
    """One reduce-scatter per step: earlier microbatches keep params gathered and gradients local."""
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils import trainer as titan_trainer
    from miles.backends.torchtitan_utils.trainer import TitanTrainer

    calls = []

    class Root(torch.nn.Module):
        def set_is_last_backward(self, value):
            calls.append(("is_last_backward", value))

        def set_reshard_after_backward(self, value):
            calls.append(("reshard_after_backward", value))

        def set_requires_gradient_sync(self, value):
            calls.append(("requires_gradient_sync", value))

        def reshard(self):
            calls.append(("reshard", None))

    monkeypatch.setattr(titan_trainer, "FSDPModule", Root)
    trainer = SimpleNamespace(
        model_parts=[Root()],
        config=SimpleNamespace(
            parallelism=SimpleNamespace(
                fsdp_defer_gradient_reduction=gather_once,
                fsdp_reshard_after_forward="never" if gather_once else "default",
            )
        ),
    )
    for is_last in (False, False, True):
        TitanTrainer._defer_gradient_reduction(trainer, is_last_microbatch=is_last)
    TitanTrainer._reshard_kept_params(trainer)
    if not gather_once:
        assert calls == []
        return
    per_microbatch = [
        [("is_last_backward", last), ("reshard_after_backward", last), ("requires_gradient_sync", last)]
        for last in (False, False, True)
    ]
    assert calls == [call for group in per_microbatch for call in group] + [("reshard", None)]


def test_the_fsdp_reduce_dtype_reaches_torchtitans_training_config(monkeypatch, tmp_path, single_gpu_dims):
    from miles.backends.torchtitan_utils.config import build_trainer_config

    monkeypatch.setattr(torch, "__version__", "2.14.0")
    with pytest.raises(ValueError, match="fsdp-reduce-dtype"):
        validate_torchtitan_args(_args(titan_fsdp_reduce_dtype="float16"))
    with pytest.raises(ValueError, match="accumulate microbatch gradients"):
        validate_torchtitan_args(_args(titan_fsdp_reduce_dtype="bfloat16", titan_fsdp_gather_once=True))
    validate_torchtitan_args(_args(titan_fsdp_reduce_dtype="bfloat16"))
    config = build_trainer_config(
        _config_args(titan_fsdp_reduce_dtype="bfloat16"),
        hf_assets_path=_checkpoint_dir(tmp_path, tie_word_embeddings=False),
        lr_total_steps=1,
        dump_subdir="x",
    )
    assert config.training.mixed_precision_reduce == "bfloat16"
