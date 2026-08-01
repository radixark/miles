"""Mock-based tests for the bridge LoRA model setup helper.

Validates that the DDP grad buffer built by ``_setup_lora_model_via_bridge``
stays in sync with ``args.use_distributed_optimizer`` — the same flag
``setup_model_and_optimizer`` uses to build the optimizer — without GPU.
"""

import sys
import types
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

_HELPERS_MODULE = "miles.backends.megatron_utils.bridge_lora_helpers"


class _RecordingDDPConfig:
    """Stand-in for megatron.bridge's DistributedDataParallelConfig."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def finalize(self):
        pass


def _stub_module(name: str, attrs: dict[str, object] | None = None, is_package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []
    for attr_name, value in (attrs or {}).items():
        setattr(module, attr_name, value)
    sys.modules[name] = module
    return module


@pytest.fixture(scope="module", autouse=True)
def _mock_megatron_bridge():
    original_modules = dict(sys.modules)
    try:
        _stub_module("megatron.bridge", {"AutoBridge": MagicMock()}, is_package=True)
        _stub_module("megatron.bridge.training", is_package=True)
        _stub_module("megatron.bridge.training.config", {"DistributedDataParallelConfig": _RecordingDDPConfig})
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def _make_args(**overrides) -> Namespace:
    args = Namespace(
        hf_checkpoint="/some/checkpoint",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        gradient_accumulation_fusion=False,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        recompute_modules=None,
        distribute_saved_activations=False,
        attention_backend=None,
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        multi_lora=False,
        target_modules=["q_proj"],
        optimizer="adam",
        use_distributed_optimizer=True,
        accumulate_allreduce_grads_in_fp32=True,
        offload_train=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _build_ddp_config(args: Namespace) -> _RecordingDDPConfig:
    """Run the helper with the bridge/LoRA/HF side effects mocked out and
    return the ``DistributedDataParallelConfig`` it handed to the provider."""
    from miles.backends.megatron_utils.bridge_lora_helpers import _setup_lora_model_via_bridge

    hf_config = MagicMock(architectures=["Qwen3ForCausalLM"])
    with (
        patch(f"{_HELPERS_MODULE}.load_hf_config", return_value=hf_config),
        patch("miles.backends.megatron_utils.lora_utils.create_lora_instance", return_value=MagicMock()),
        patch(
            "miles.backends.megatron_utils.multi_lora_utils.create_multi_lora_instance",
            return_value=MagicMock(),
        ),
    ):
        _setup_lora_model_via_bridge(args)

    bridge = sys.modules["megatron.bridge"].AutoBridge.from_hf_pretrained.return_value
    provider = bridge.to_megatron_provider.return_value
    return provider.provide_distributed_model.call_args.kwargs["ddp_config"]


class TestBridgeLoraDDPConfig:
    """The grad buffer layout must follow args, not a locally re-derived rule."""

    def test_distributed_optimizer_grad_buffer(self):
        ddp_config = _build_ddp_config(_make_args(use_distributed_optimizer=True))
        assert ddp_config.use_distributed_optimizer is True

    def test_non_distributed_optimizer_grad_buffer(self):
        # A non-distributed optimizer steps on whole params, so the grad buffer
        # must all-reduce; a reduce-scatter buffer would leave each rank with
        # only its own shard reduced at DP > 1.
        ddp_config = _build_ddp_config(_make_args(optimizer="sgd", use_distributed_optimizer=False))
        assert ddp_config.use_distributed_optimizer is False

    def test_multi_lora_grad_buffer(self):
        ddp_config = _build_ddp_config(_make_args(multi_lora=True, use_distributed_optimizer=False))
        assert ddp_config.use_distributed_optimizer is False

    def test_grad_reduce_in_fp32_follows_args(self):
        ddp_config = _build_ddp_config(_make_args(accumulate_allreduce_grads_in_fp32=False))
        assert ddp_config.grad_reduce_in_fp32 is False
