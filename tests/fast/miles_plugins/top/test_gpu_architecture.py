from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sglang.srt.true_on_policy.contracts import TRUE_ON_POLICY_V1
from sglang.srt.true_on_policy.device import DEVICE_STATE_KEY

from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.true_on_policy import device
from miles.true_on_policy.device import gpu_architecture_mismatch, refuse_mixed_gpu_architectures
from miles_plugins.top import binding

H200 = {"capability": [9, 0], "name": "NVIDIA H200"}
H100 = {"capability": [9, 0], "name": "NVIDIA H100 80GB HBM3"}
B200 = {"capability": [10, 0], "name": "NVIDIA B200"}


def _server_info(*dp_rank_devices):
    """/server_info with one internal state per DP rank; None leaves the device report out."""
    return {
        "internal_states": [
            {"true_on_policy_contract": TRUE_ON_POLICY_V1} | ({} if d is None else {DEVICE_STATE_KEY: d})
            for d in dp_rank_devices
        ]
    }


def _engine(*dp_rank_devices):
    return SimpleNamespace(get_server_info=AsyncMock(return_value=_server_info(*dp_rank_devices)))


@pytest.fixture
def trainer_on_h200():
    with patch.object(device, "_trainer_device", H200):
        yield


def test_engines_of_the_trainer_architecture_pass():
    """Same SM, another product name: the per-architecture kernels are the same, so the run is admitted."""
    infos = [_server_info(H200), _server_info(H100, H200)]
    assert gpu_architecture_mismatch(trainer=H200, engine_cell_ids=["c0", "c1"], server_infos=infos) is None


def test_an_engine_on_another_architecture_is_refused():
    error = gpu_architecture_mismatch(
        trainer=H200, engine_cell_ids=["c0", "c1"], server_infos=[_server_info(H200), _server_info(B200)]
    )
    assert error.startswith("[top] engine c1 runs SM100 (NVIDIA B200), the trainer SM90 (NVIDIA H200)")
    assert "c0" not in error


def test_a_mismatched_dp_rank_is_named():
    error = gpu_architecture_mismatch(trainer=H200, engine_cell_ids=["c0"], server_infos=[_server_info(H200, B200)])
    assert error.startswith("[top] engine c0 (DP rank 1) runs SM100")


def test_an_engine_without_a_device_report_is_refused():
    """An engine off the contract, or on an SGLang without the report, must not pass by default."""
    error = gpu_architecture_mismatch(trainer=H200, engine_cell_ids=["c0"], server_infos=[_server_info(None)])
    assert error.startswith("[top] engine c0 reports no GPU architecture")


def test_an_unpinned_trainer_is_refused():
    error = gpu_architecture_mismatch(trainer=None, engine_cell_ids=["c0"], server_infos=[_server_info(H200)])
    assert "before it pinned SGLang" in error


def test_refusal_reads_every_engine(trainer_on_h200):
    engines = [_engine(H200), _engine(B200)]
    with pytest.raises(RuntimeError, match=r"engine c1 runs SM100 \(NVIDIA B200\), the trainer SM90"):
        refuse_mixed_gpu_architectures(rollout_engines=engines, engine_cell_ids=["c0", "c1"])
    for engine in engines:
        engine.get_server_info.assert_awaited_once()


def test_an_unreachable_engine_raises_its_own_error(trainer_on_h200):
    """Rank 0 hands the error to every rank, so it must not be swallowed into a pass."""
    engine = SimpleNamespace(get_server_info=AsyncMock(side_effect=ConnectionError("engine down")))
    with pytest.raises(ConnectionError, match="engine down"):
        refuse_mixed_gpu_architectures(rollout_engines=[engine], engine_cell_ids=["c0"])


def test_every_rank_raises_rank_zeros_refusal(trainer_on_h200):
    """Only rank 0 reads the engines; the others take its verdict from the broadcast instead of moving on."""

    def broadcast(objects, src, group):
        objects[0] = RuntimeError("[top] engine c0 runs SM100")

    with (
        patch.object(device.dist, "is_initialized", return_value=True),
        patch.object(device.dist, "get_rank", return_value=1),
        patch.object(device.dist, "broadcast_object_list", side_effect=broadcast),
        patch.object(device, "get_gloo_group", return_value=MagicMock()),
    ):
        engine = _engine(B200)
        with pytest.raises(RuntimeError, match="engine c0 runs SM100"):
            refuse_mixed_gpu_architectures(rollout_engines=[engine], engine_cell_ids=["c0"])
    engine.get_server_info.assert_not_awaited()


def _updater(true_on_policy_mode):
    protocol = MagicMock(spec=WeightTransferProtocol, supports_lora=False, is_sender=True)
    with patch(
        "miles.backends.training_utils.weight_update.updater.get_weight_transfer_protocol", return_value=protocol
    ):
        return WeightUpdater(
            Namespace(
                true_on_policy_mode=true_on_policy_mode,
                enable_mtp_training=False,
                sglang=SimpleNamespace(base_args={"speculative_algorithm": None}),
            ),
            [],
            weights_getter=lambda: {},
            model_name="test-model",
            quantization_config=None,
            iterator_factory=lambda *a, **k: MagicMock(),
            parallel_state=MagicMock(),
            is_lora=False,
        )


def _connect(updater, engines):
    with patch("miles.backends.training_utils.weight_update.updater.weight_update_selector", return_value="all"):
        updater.connect_rollout_engines(
            engines, engine_gpu_counts=None, engine_gpu_offsets=None, engine_cell_ids=["c0"]
        )


def test_top_connect_refuses_before_the_transport_connects(trainer_on_h200):
    """Connect runs on first contact and every reconnect, so a late engine on another GPU is refused too."""
    updater = _updater(true_on_policy_mode=True)
    with pytest.raises(RuntimeError, match="engine c0 runs SM100"):
        _connect(updater, [_engine(B200)])
    updater.protocol.connect.assert_not_called()

    _connect(updater, [_engine(H100)])
    updater.protocol.connect.assert_called_once()


def test_connect_without_top_reads_no_server_info():
    updater = _updater(true_on_policy_mode=False)
    engine = _engine(B200)
    _connect(updater, [engine])
    engine.get_server_info.assert_not_awaited()
    updater.protocol.connect.assert_called_once()


def test_pin_records_the_trainer_device():
    """The trainer's side of the comparison is taken where it pins SGLang, on its own GPU."""
    pinned = SimpleNamespace(
        true_on_policy_contract=TRUE_ON_POLICY_V1, enable_deterministic_inference=True, model_path="/ckpt"
    )
    args = SimpleNamespace(true_on_policy_contract=TRUE_ON_POLICY_V1, hf_checkpoint="/ckpt")
    with (
        patch.object(device, "_trainer_device", None),
        patch.object(device, "device_report", return_value=B200),
        patch.object(binding, "_existing_server_args", return_value=pinned),
        patch("sglang.srt.runtime_context.publish_role", return_value="top_trainer"),
    ):
        assert binding.pin_sglang(args, {}) == TRUE_ON_POLICY_V1
        assert device._trainer_device == B200
