from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.utils.profile_utils import (
    _create_trace_handler,
    _summarize_low_precision_events,
    categorize_low_precision_kernel,
)


@pytest.mark.parametrize(
    ("kernel_name", "expected"),
    [
        ("transformer_engine::quantize_mxfp8_kernel", "quantization"),
        ("transformer_engine::dequantize_kernel", "dequantization"),
        ("nvjet_sm100_qqtst_gemm", "gemm"),
        ("transformer_engine::compute_scale_kernel", "scale_amax"),
        ("transformer_engine::swizzle_row_scaling", "layout_memory"),
        ("unrelated_activation_kernel", None),
    ],
)
def test_categorize_low_precision_kernel(kernel_name, expected):
    assert categorize_low_precision_kernel(kernel_name) == expected


def test_summarize_low_precision_events_preserves_overlap_semantics():
    cuda = SimpleNamespace(name="CUDA")
    events = [
        SimpleNamespace(
            name="_Linear",
            input_shapes=[[256, 1024], [2048, 1024]],
            kernels=[
                SimpleNamespace(name="quantize_mxfp8_kernel", duration=3.5),
                SimpleNamespace(name="nvjet_sm100_qqtst", duration=7.0),
            ],
            device_type=SimpleNamespace(name="CPU"),
        ),
        SimpleNamespace(
            name="aten::gelu",
            input_shapes=[[256, 2048]],
            kernels=[SimpleNamespace(name="activation_kernel", duration=2.0)],
            device_type=SimpleNamespace(name="CPU"),
        ),
        SimpleNamespace(name="quantize_mxfp8_kernel", device_type=cuda, device_time=3.5, kernels=[]),
        SimpleNamespace(name="nvjet_sm100_qqtst", device_type=cuda, device_time=7.0, kernels=[]),
        SimpleNamespace(name="activation_kernel", device_type=cuda, device_time=2.0, kernels=[]),
    ]

    summary = _summarize_low_precision_events(events, name="train_actor", rank=2, step=11)

    assert summary["schema_version"] == 1
    assert summary["profile_name"] == "train_actor"
    assert summary["rank"] == 2
    assert summary["step"] == 11
    assert summary["categories"]["quantization"] == {"duration_us": 3.5, "kernel_calls": 1}
    assert summary["categories"]["gemm"] == {"duration_us": 7.0, "kernel_calls": 1}
    assert summary["uncategorized"] == {"duration_us": 2.0, "kernel_calls": 1}
    assert summary["kernels"][0]["input_shapes"] == [[[256, 1024], [2048, 1024]]]


def test_summarize_low_precision_events_does_not_count_parent_kernel_links():
    linked_kernel = SimpleNamespace(name="quantize_mxfp8_kernel", duration=4.0)
    events = [
        SimpleNamespace(
            name="_Linear",
            input_shapes=[[8, 16]],
            kernels=[linked_kernel],
            device_type=SimpleNamespace(name="CPU"),
        ),
        SimpleNamespace(
            name="Runtime Triggered Module Loading",
            input_shapes=[],
            kernels=[linked_kernel],
            device_type=SimpleNamespace(name="CPU"),
        ),
        SimpleNamespace(
            name="quantize_mxfp8_kernel",
            device_type=SimpleNamespace(name="CUDA"),
            device_time=4.0,
            kernels=[],
        ),
    ]

    summary = _summarize_low_precision_events(events, name="train_overall", rank=0, step=1)

    assert summary["categories"]["quantization"] == {"duration_us": 4.0, "kernel_calls": 1}
    assert summary["kernels"][0]["parents"] == ["Runtime Triggered Module Loading", "_Linear"]


def test_create_trace_handler_keeps_disabled_path_unwrapped():
    args = SimpleNamespace(tensorboard_dir="/tmp/traces", profile_low_precision=False)
    tensorboard_handler = object()
    with (
        patch("miles.utils.profile_utils.torch.distributed.get_rank", return_value=3),
        patch(
            "miles.utils.profile_utils.torch.profiler.tensorboard_trace_handler",
            return_value=tensorboard_handler,
        ) as create_tensorboard_handler,
    ):
        handler = _create_trace_handler(args, name="train_overall")

    assert handler is tensorboard_handler
    create_tensorboard_handler.assert_called_once_with(
        "/tmp/traces",
        worker_name="train_overall_rank_3",
        use_gzip=True,
    )


def test_create_trace_handler_requires_output_directory_for_low_precision_summary():
    args = SimpleNamespace(tensorboard_dir=None, profile_low_precision=True)
    with (
        patch("miles.utils.profile_utils.torch.distributed.get_rank", return_value=0),
        pytest.raises(ValueError, match="--profile-low-precision requires --tensorboard-dir"),
    ):
        _create_trace_handler(args, name="train_overall")
