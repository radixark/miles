import importlib
import sys
import types
from types import SimpleNamespace


class _RecordingScheduler:
    def __init__(self, optimizer, **kwargs):
        assert kwargs["lr_decay_steps"] > 0
        self.kwargs = kwargs


def _register(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _load_model_module(monkeypatch):
    stub = object

    _register(monkeypatch, "megatron")
    _register(monkeypatch, "megatron.core", mpu=types.ModuleType("megatron.core.mpu"))
    _register(monkeypatch, "megatron.core.mpu")
    _register(monkeypatch, "megatron.core.distributed", DistributedDataParallel=stub, finalize_model_grads=stub)
    _register(monkeypatch, "megatron.core.enums", ModelType=stub)
    _register(monkeypatch, "megatron.core.models", gpt=types.ModuleType("megatron.core.models.gpt"))
    _register(monkeypatch, "megatron.core.models.gpt", GPTModel=stub)
    _register(monkeypatch, "megatron.core.optimizer", OptimizerConfig=stub, get_megatron_optimizer=stub)
    _register(monkeypatch, "megatron.core.optimizer.muon", get_megatron_muon_optimizer=stub)
    _register(monkeypatch, "megatron.core.optimizer.optimizer", MegatronOptimizer=stub)
    _register(monkeypatch, "megatron.core.optimizer_param_scheduler", OptimizerParamScheduler=_RecordingScheduler)
    _register(monkeypatch, "megatron.core.pipeline_parallel", get_forward_backward_func=stub)
    _register(monkeypatch, "megatron.core.utils", get_model_config=stub)
    _register(monkeypatch, "megatron.training")
    _register(monkeypatch, "megatron.training.global_vars", get_args=stub)
    _register(monkeypatch, "megatron.training.training", get_model=stub)

    _register(monkeypatch, "miles.utils.dumper_utils", DumperMegatronUtil=stub, DumperPhase=stub)
    _register(monkeypatch, "miles.utils.memory_utils", clear_memory=stub)
    _register(monkeypatch, "miles.backends.training_utils.ci_utils", check_grad_norm=stub, check_kl=stub)
    _register(monkeypatch, "miles.backends.training_utils.data", DataIterator=stub, get_batch=stub)
    _register(
        monkeypatch,
        "miles.backends.training_utils.log_utils",
        aggregate_forward_results=stub,
        aggregate_train_losses=stub,
        log_train_step=stub,
    )
    _register(monkeypatch, "miles.backends.training_utils.loss", loss_function=stub)
    _register(monkeypatch, "miles.backends.training_utils.parallel", get_parallel_state=stub)
    _register(
        monkeypatch,
        "miles.backends.megatron_utils.checkpoint",
        load_checkpoint=stub,
        save_checkpoint=stub,
        save_checkpoint_with_lora=stub,
    )
    _register(
        monkeypatch,
        "miles.backends.megatron_utils.ci_utils",
        check_model_hashes=stub,
        check_peak_gpu_memory_after_load=stub,
        compute_model_hashes_by_layer=stub,
        save_model_hashes=stub,
    )
    _register(monkeypatch, "miles.backends.megatron_utils.initialize", is_megatron_main_rank=stub)
    _register(
        monkeypatch,
        "miles.backends.megatron_utils.lora_utils",
        is_lora_enabled=stub,
        is_lora_model=stub,
        save_lora_checkpoint=stub,
    )
    _register(
        monkeypatch,
        "miles.backends.megatron_utils.bridge_lora_helpers",
        _ensure_model_list=stub,
        _setup_lora_model_via_bridge=stub,
    )
    _register(monkeypatch, "miles.backends.megatron_utils.model_provider", get_model_provider_func=stub)
    _register(monkeypatch, "miles.backends.megatron_utils.parallel", get_packed_seq_params=stub)

    sys.modules.pop("miles.backends.megatron_utils.model", None)
    return importlib.import_module("miles.backends.megatron_utils.model")


def _make_args(**overrides):
    args = SimpleNamespace(
        num_rollout=4,
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        global_batch_size=16,
        lr_decay_iters=None,
        lr_wsd_decay_iters=None,
        lr_warmup_fraction=None,
        lr_warmup_iters=0,
        lr_warmup_init=0.0,
        lr=1e-6,
        min_lr=0.0,
        lr_decay_style="constant",
        start_weight_decay=0.0,
        end_weight_decay=0.0,
        weight_decay_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=False,
        lr_wsd_decay_style="exponential",
    )
    args.__dict__.update(overrides)
    return args


def test_eval_only_num_rollout_zero_does_not_crash(monkeypatch):
    model = _load_model_module(monkeypatch)
    args = _make_args(num_rollout=0)

    model.get_optimizer_param_scheduler(args, optimizer=object())

    assert args.train_iters == 1


def test_train_iters_clamp_is_noop_for_normal_training(monkeypatch):
    model = _load_model_module(monkeypatch)
    args = _make_args(num_rollout=4, rollout_batch_size=8, n_samples_per_prompt=8, global_batch_size=16)

    model.get_optimizer_param_scheduler(args, optimizer=object())

    assert args.train_iters == 16
