import argparse
import importlib
import sys
import types
from argparse import Namespace

import pytest


def _install_argument_test_stubs():
    if "sglang_router.launch_router" not in sys.modules:
        launch_router_module = types.ModuleType("sglang_router.launch_router")

        class RouterArgs:
            @staticmethod
            def add_cli_args(parser, **kwargs):
                return parser

        launch_router_module.RouterArgs = RouterArgs
        router_module = types.ModuleType("sglang_router")
        router_module.launch_router = launch_router_module
        sys.modules["sglang_router"] = router_module
        sys.modules["sglang_router.launch_router"] = launch_router_module

    if "transformers" not in sys.modules:
        transformers_module = types.ModuleType("transformers")

        class AutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("AutoConfig.from_pretrained should not be called in predictive argument tests.")

        transformers_module.AutoConfig = AutoConfig
        sys.modules["transformers"] = transformers_module

    if "ray" not in sys.modules:
        ray_module = types.ModuleType("ray")

        def remote(*args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        ray_module.remote = remote
        ray_module.init = lambda *args, **kwargs: None
        ray_module.shutdown = lambda *args, **kwargs: None
        ray_module.get = lambda refs: refs
        ray_module.nodes = lambda: []
        ray_private_module = types.ModuleType("ray._private")
        services_module = types.ModuleType("ray._private.services")
        services_module.get_node_ip_address = lambda: "127.0.0.1"
        ray_private_module.services = services_module
        ray_module._private = ray_private_module
        sys.modules["ray"] = ray_module
        sys.modules["ray._private"] = ray_private_module
        sys.modules["ray._private.services"] = services_module

        ray_util_module = types.ModuleType("ray.util")
        scheduling_module = types.ModuleType("ray.util.scheduling_strategies")

        class NodeAffinitySchedulingStrategy:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        scheduling_module.NodeAffinitySchedulingStrategy = NodeAffinitySchedulingStrategy
        ray_util_module.scheduling_strategies = scheduling_module
        sys.modules["ray.util"] = ray_util_module
        sys.modules["ray.util.scheduling_strategies"] = scheduling_module

    if "miles.backends.sglang_utils.arguments" not in sys.modules:
        sglang_args_module = types.ModuleType("miles.backends.sglang_utils.arguments")
        sglang_args_module.add_sglang_arguments = lambda parser: parser
        sglang_args_module.validate_args = lambda args: None
        sys.modules["miles.backends.sglang_utils.arguments"] = sglang_args_module

    if "httpx" not in sys.modules:
        httpx_module = types.ModuleType("httpx")

        class Limits:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class Timeout:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class Client:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class AsyncClient(Client):
            async def get(self, *args, **kwargs):
                raise RuntimeError("httpx.AsyncClient.get should not be called in predictive argument tests.")

            async def post(self, *args, **kwargs):
                raise RuntimeError("httpx.AsyncClient.post should not be called in predictive argument tests.")

            async def delete(self, *args, **kwargs):
                raise RuntimeError("httpx.AsyncClient.delete should not be called in predictive argument tests.")

        class HTTPStatusError(Exception):
            def __init__(self, *args, response=None, **kwargs):
                super().__init__(*args)
                self.response = response

        httpx_module.Limits = Limits
        httpx_module.Timeout = Timeout
        httpx_module.Client = Client
        httpx_module.AsyncClient = AsyncClient
        httpx_module.HTTPStatusError = HTTPStatusError
        sys.modules["httpx"] = httpx_module


_install_argument_test_stubs()

arguments = importlib.import_module("miles.utils.arguments")
PREDICTIVE_ROUTING_REPLAY_LOSS_TYPES = arguments.PREDICTIVE_ROUTING_REPLAY_LOSS_TYPES
PREDICTIVE_ROUTING_REPLAY_LAYER_SCALE_SCHEDULES = arguments.PREDICTIVE_ROUTING_REPLAY_LAYER_SCALE_SCHEDULES
PREDICTIVE_ROUTING_REPLAY_STORAGE_DTYPES = arguments.PREDICTIVE_ROUTING_REPLAY_STORAGE_DTYPES
PREDICTIVE_HIDDEN_SHIFT_WEIGHT_MODES = arguments.PREDICTIVE_HIDDEN_SHIFT_WEIGHT_MODES
_validate_predictive_routing_replay_args = arguments._validate_predictive_routing_replay_args
_validate_router_logits_args = arguments._validate_router_logits_args
get_miles_extra_args_provider = arguments.get_miles_extra_args_provider


def _make_parser():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    return parser


def _make_validation_args(**overrides):
    values = {
        "enable_predictive_routing_replay": False,
        "bias_predictor_loss_type": "kl-post",
        "bias_predictor_lr_mult": 1000.0,
        "predictive_downsample_batch_size": None,
        "predictive_downsample_max_len_limit": None,
        "predictive_max_total_tokens": None,
        "predictive_max_hidden_shift_relative_norm": None,
        "predictive_hidden_shift_weight_mode": "binary",
        "predictive_boundary_loss_max_weight": None,
        "predictive_boundary_loss_min_margin": 1e-4,
        "predictive_min_post_topk_margin_for_flip": None,
        "predictive_layer_scale_schedule": "none",
        "predictive_layer_scale_min": 1.0,
        "predictive_max_delta_to_old_ratio": None,
        "predictive_max_delta_to_topk_margin_ratio": None,
        "predictive_max_delta_to_topk_margin_ratio_final": None,
        "predictive_topk_margin_ratio_anneal_start_rollout": None,
        "predictive_topk_margin_ratio_anneal_end_rollout": None,
        "predictive_storage_dtype": "bf16",
        "train_backend": "megatron",
        "use_routing_replay": False,
        "use_rollout_routing_replay": False,
        "allgather_cp": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_predictive_flags_parse():
    parser = _make_parser()
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "64",
            "--enable-predictive-routing-replay",
            "--bias-predictor-loss-type",
            "kl-post",
            "--bias-predictor-lr-mult",
            "321.0",
            "--predictive-downsample-batch-size",
            "4",
            "--predictive-downsample-max-len-limit",
            "1024",
            "--predictive-max-total-tokens",
            "2048",
            "--predictive-max-hidden-shift-relative-norm",
            "0.02",
            "--predictive-hidden-shift-weight-mode",
            "quadratic",
            "--predictive-boundary-loss-max-weight",
            "4.0",
            "--predictive-boundary-loss-min-margin",
            "0.001",
            "--predictive-min-post-topk-margin-for-flip",
            "0.05",
            "--predictive-layer-scale-schedule",
            "sqrt_decay",
            "--predictive-layer-scale-min",
            "0.5",
            "--predictive-max-delta-to-old-ratio",
            "0.02",
            "--predictive-max-delta-to-topk-margin-ratio",
            "1.0",
            "--predictive-max-delta-to-topk-margin-ratio-final",
            "2.0",
            "--predictive-topk-margin-ratio-anneal-start-rollout",
            "80",
            "--predictive-topk-margin-ratio-anneal-end-rollout",
            "160",
            "--predictive-storage-dtype",
            "fp16",
        ]
    )

    assert args.enable_predictive_routing_replay is True
    assert args.bias_predictor_loss_type == "kl-post"
    assert args.bias_predictor_lr_mult == pytest.approx(321.0)
    assert args.predictive_downsample_batch_size == 4
    assert args.predictive_downsample_max_len_limit == 1024
    assert args.predictive_max_total_tokens == 2048
    assert args.predictive_max_hidden_shift_relative_norm == pytest.approx(0.02)
    assert args.predictive_hidden_shift_weight_mode == "quadratic"
    assert args.predictive_boundary_loss_max_weight == pytest.approx(4.0)
    assert args.predictive_boundary_loss_min_margin == pytest.approx(0.001)
    assert args.predictive_min_post_topk_margin_for_flip == pytest.approx(0.05)
    assert args.predictive_layer_scale_schedule == "sqrt_decay"
    assert args.predictive_layer_scale_min == pytest.approx(0.5)
    assert args.predictive_max_delta_to_old_ratio == pytest.approx(0.02)
    assert args.predictive_max_delta_to_topk_margin_ratio == pytest.approx(1.0)
    assert args.predictive_max_delta_to_topk_margin_ratio_final == pytest.approx(2.0)
    assert args.predictive_topk_margin_ratio_anneal_start_rollout == 80
    assert args.predictive_topk_margin_ratio_anneal_end_rollout == 160
    assert args.predictive_storage_dtype == "fp16"


def test_router_logits_flags_parse():
    parser = _make_parser()
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "64",
            "--router-logits-path",
            "/tmp/router_logits",
            "--router-logits-save-freq",
            "10",
        ]
    )

    assert args.router_logits_path == "/tmp/router_logits"
    assert args.router_logits_save_freq == 10


def test_predictive_loss_type_defaults_to_kl_post():
    parser = _make_parser()
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "64",
        ]
    )

    assert args.bias_predictor_loss_type == "kl-post"


def test_predictive_validation_sets_aliases():
    args = _make_validation_args(enable_predictive_routing_replay=True, use_routing_replay=True)

    _validate_predictive_routing_replay_args(args)

    assert args.enable_bias_predictor is True
    assert args.predictive_routing_replay_mode == "R2"


def test_predictive_layer_scale_schedule_choices_exported():
    assert PREDICTIVE_ROUTING_REPLAY_LAYER_SCALE_SCHEDULES == ("none", "linear_decay", "sqrt_decay", "cosine_decay")


def test_predictive_hidden_shift_weight_mode_choices_exported():
    assert PREDICTIVE_HIDDEN_SHIFT_WEIGHT_MODES == ("binary", "linear", "quadratic")


def test_predictive_validation_rejects_invalid_stabilizer_args():
    with pytest.raises(AssertionError, match="predictive-max-hidden-shift-relative-norm"):
        _validate_predictive_routing_replay_args(
            _make_validation_args(
                enable_predictive_routing_replay=True,
                use_routing_replay=True,
                predictive_max_hidden_shift_relative_norm=0.0,
            )
        )

    with pytest.raises(AssertionError, match="predictive hidden-shift weight mode"):
        _validate_predictive_routing_replay_args(
            _make_validation_args(
                enable_predictive_routing_replay=True,
                use_routing_replay=True,
                predictive_hidden_shift_weight_mode="bad-mode",
            )
        )

    with pytest.raises(AssertionError, match="predictive-layer-scale-min"):
        _validate_predictive_routing_replay_args(
            _make_validation_args(
                enable_predictive_routing_replay=True,
                use_routing_replay=True,
                predictive_layer_scale_min=0.0,
            )
        )

    with pytest.raises(AssertionError, match="predictive-max-delta-to-old-ratio"):
        _validate_predictive_routing_replay_args(
            _make_validation_args(
                enable_predictive_routing_replay=True,
                use_routing_replay=True,
                predictive_max_delta_to_old_ratio=0.0,
            )
        )

    with pytest.raises(AssertionError, match="predictive-min-post-topk-margin-for-flip"):
        _validate_predictive_routing_replay_args(
            _make_validation_args(
                enable_predictive_routing_replay=True,
                use_routing_replay=True,
                predictive_min_post_topk_margin_for_flip=0.0,
            )
        )


def test_predictive_validation_disabled_path_sets_aliases():
    args = _make_validation_args()

    _validate_predictive_routing_replay_args(args)

    assert args.enable_bias_predictor is False
    assert args.predictive_routing_replay_mode is None


def test_predictive_validation_requires_routing_replay():
    args = _make_validation_args(enable_predictive_routing_replay=True)

    with pytest.raises(AssertionError, match="requires --use-routing-replay"):
        _validate_predictive_routing_replay_args(args)


def test_predictive_validation_rejects_rollout_routing_replay():
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        use_rollout_routing_replay=True,
    )

    with pytest.raises(AssertionError, match="actor-side R2"):
        _validate_predictive_routing_replay_args(args)


def test_predictive_validation_requires_megatron():
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        train_backend="fsdp",
    )

    with pytest.raises(AssertionError, match="megatron backend"):
        _validate_predictive_routing_replay_args(args)


def test_predictive_validation_rejects_allgather_cp():
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        allgather_cp=True,
    )

    with pytest.raises(AssertionError, match="allgather-cp"):
        _validate_predictive_routing_replay_args(args)


@pytest.mark.parametrize("loss_type", PREDICTIVE_ROUTING_REPLAY_LOSS_TYPES)
def test_predictive_validation_accepts_supported_loss_types(loss_type):
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        bias_predictor_loss_type=loss_type,
    )

    _validate_predictive_routing_replay_args(args)


@pytest.mark.parametrize("storage_dtype", PREDICTIVE_ROUTING_REPLAY_STORAGE_DTYPES)
def test_predictive_validation_accepts_supported_storage_dtypes(storage_dtype):
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        predictive_storage_dtype=storage_dtype,
    )

    _validate_predictive_routing_replay_args(args)


def test_predictive_validation_accepts_zero_lr_multiplier():
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        bias_predictor_lr_mult=0,
    )

    _validate_predictive_routing_replay_args(args)


def test_predictive_validation_rejects_negative_lr_multiplier():
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        bias_predictor_lr_mult=-1,
    )

    with pytest.raises(AssertionError, match="bias-predictor-lr-mult"):
        _validate_predictive_routing_replay_args(args)


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("predictive_downsample_batch_size", 0, "predictive-downsample-batch-size"),
        ("predictive_downsample_max_len_limit", 0, "predictive-downsample-max-len-limit"),
        ("predictive_max_total_tokens", 0, "predictive-max-total-tokens"),
        ("predictive_max_delta_to_topk_margin_ratio", 0, "predictive-max-delta-to-topk-margin-ratio"),
    ],
)
def test_predictive_validation_rejects_nonpositive_downsample_values(field_name, field_value, message):
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        **{field_name: field_value},
    )

    with pytest.raises(AssertionError, match=message):
        _validate_predictive_routing_replay_args(args)


def test_predictive_validation_accepts_topk_margin_ratio_above_one():
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        predictive_max_delta_to_topk_margin_ratio=1.25,
    )

    _validate_predictive_routing_replay_args(args)


def test_predictive_validation_rejects_incomplete_topk_margin_ratio_annealing():
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        predictive_max_delta_to_topk_margin_ratio=1.0,
        predictive_max_delta_to_topk_margin_ratio_final=2.0,
    )

    with pytest.raises(AssertionError, match="predictive-topk-margin-ratio-anneal-end-rollout"):
        _validate_predictive_routing_replay_args(args)


def test_predictive_validation_rejects_topk_margin_ratio_anneal_without_final_ratio():
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        predictive_max_delta_to_topk_margin_ratio=1.0,
        predictive_topk_margin_ratio_anneal_end_rollout=160,
    )

    with pytest.raises(AssertionError, match="top-k margin-ratio anneal rollout arguments"):
        _validate_predictive_routing_replay_args(args)


def test_predictive_validation_rejects_topk_margin_ratio_anneal_end_before_start():
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        predictive_max_delta_to_topk_margin_ratio=1.0,
        predictive_max_delta_to_topk_margin_ratio_final=2.0,
        predictive_topk_margin_ratio_anneal_start_rollout=100,
        predictive_topk_margin_ratio_anneal_end_rollout=80,
    )

    with pytest.raises(AssertionError, match="anneal-end-rollout must be greater"):
        _validate_predictive_routing_replay_args(args)


def test_router_logits_validation_normalizes_empty_path():
    args = Namespace(router_logits_path="", router_logits_save_freq=1)

    _validate_router_logits_args(args)

    assert args.router_logits_path is None


def test_router_logits_validation_rejects_nonpositive_save_frequency():
    args = Namespace(router_logits_path="/tmp/router_logits", router_logits_save_freq=0)

    with pytest.raises(AssertionError, match="router-logits-save-freq"):
        _validate_router_logits_args(args)
