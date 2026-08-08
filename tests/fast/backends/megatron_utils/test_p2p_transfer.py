import importlib
import sys
import types
from concurrent.futures import Future
from contextlib import nullcontext
from threading import Event, Thread
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest


P2P_MODULE_NAME = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p"
TRANSFER_UTILS_MODULE_NAME = (
    "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils"
)
TRANSFER_WAIT_FAILURE_MESSAGE = "P2P weight transfer failed; see trainer logs for details."
WEIGHT_UPDATE_FAILURE_MESSAGE = (
    "P2P weight transfer failed on at least one trainer rank; rollout engines were not finalized or resumed."
)
TRAINER_RECREATE_MESSAGE = (
    "P2P weight transfer state cannot be reused after a failure; the trainer actor must be recreated."
)
MISSING_MODULE_ATTRIBUTE = object()


def _remove_module(module_name: str) -> tuple[ModuleType | None, object]:
    parent_name, _, attribute_name = module_name.rpartition(".")
    parent_module = sys.modules.get(parent_name)
    saved_attribute = (
        getattr(parent_module, attribute_name, MISSING_MODULE_ATTRIBUTE)
        if parent_module is not None
        else MISSING_MODULE_ATTRIBUTE
    )
    return sys.modules.pop(module_name, None), saved_attribute


def _restore_module(module_name: str, saved_module: ModuleType | None, saved_attribute: object) -> None:
    if saved_module is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = saved_module

    parent_name, _, attribute_name = module_name.rpartition(".")
    parent_module = sys.modules.get(parent_name)
    if parent_module is None:
        return
    if saved_attribute is MISSING_MODULE_ATTRIBUTE:
        parent_module.__dict__.pop(attribute_name, None)
    else:
        setattr(parent_module, attribute_name, saved_attribute)


def _stub_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    *,
    package: bool = False,
) -> ModuleType:
    module = types.ModuleType(module_name)
    if package:
        module.__path__ = []
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def _install_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    mooncake = types.ModuleType("mooncake")
    mooncake.__path__ = []
    mooncake_engine = types.ModuleType("mooncake.engine")
    mooncake_engine.TransferEngine = object
    mooncake.engine = mooncake_engine
    monkeypatch.setitem(sys.modules, "mooncake", mooncake)
    monkeypatch.setitem(sys.modules, "mooncake.engine", mooncake_engine)

    sglang = _stub_module(monkeypatch, "sglang", package=True)
    sglang_srt = _stub_module(monkeypatch, "sglang.srt", package=True)
    sglang.srt = sglang_srt

    server_args = _stub_module(monkeypatch, "sglang.srt.server_args")
    server_args.ServerArgs = type("ServerArgs", (), {})
    server_args.set_global_server_args_for_scheduler = lambda *_args, **_kwargs: None
    sglang_srt.server_args = server_args

    configs = _stub_module(monkeypatch, "sglang.srt.configs", package=True)
    sglang_srt.configs = configs
    device_config = _stub_module(monkeypatch, "sglang.srt.configs.device_config")
    device_config.DeviceConfig = type("DeviceConfig", (), {})
    load_config = _stub_module(monkeypatch, "sglang.srt.configs.load_config")
    load_config.LoadConfig = type("LoadConfig", (), {})
    model_config = _stub_module(monkeypatch, "sglang.srt.configs.model_config")
    model_config.ModelConfig = type("ModelConfig", (), {})
    configs.device_config = device_config
    configs.load_config = load_config
    configs.model_config = model_config

    distributed = _stub_module(monkeypatch, "sglang.srt.distributed", package=True)
    parallel_state = _stub_module(monkeypatch, "sglang.srt.distributed.parallel_state")
    parallel_state.ParallelismContext = type("ParallelismContext", (), {})
    parallel_state.RankParallelismConfig = type("RankParallelismConfig", (), {})
    distributed.parallel_state = parallel_state
    sglang_srt.distributed = distributed

    layers = _stub_module(monkeypatch, "sglang.srt.layers", package=True)
    moe = _stub_module(monkeypatch, "sglang.srt.layers.moe")
    moe.initialize_moe_config = lambda *_args, **_kwargs: None
    quantization = _stub_module(monkeypatch, "sglang.srt.layers.quantization", package=True)
    fp4_utils = _stub_module(monkeypatch, "sglang.srt.layers.quantization.fp4_utils")
    fp4_utils.initialize_fp4_gemm_config = lambda *_args, **_kwargs: None
    fp8_utils = _stub_module(monkeypatch, "sglang.srt.layers.quantization.fp8_utils")
    fp8_utils.initialize_fp8_gemm_config = lambda *_args, **_kwargs: None
    fp8_utils.mxfp8_group_quantize = lambda *_args, **_kwargs: None
    fp8_utils.per_block_cast_to_fp8 = lambda *_args, **_kwargs: None
    fp8_utils.quant_weight_ue8m0 = lambda *_args, **_kwargs: None
    fp8_utils.transform_scale_ue8m0 = lambda value, **_kwargs: value
    quantization.fp4_utils = fp4_utils
    quantization.fp8_utils = fp8_utils
    layers.moe = moe
    layers.quantization = quantization
    sglang_srt.layers = layers

    model_loader = _stub_module(monkeypatch, "sglang.srt.model_loader", package=True)
    model_loader.get_model = lambda *_args, **_kwargs: None
    parameter_mapper = _stub_module(monkeypatch, "sglang.srt.model_loader.parameter_mapper")
    parameter_mapper.ParameterMapper = type("ParameterMapper", (), {})
    model_loader_utils = _stub_module(monkeypatch, "sglang.srt.model_loader.utils")
    model_loader_utils.should_deepgemm_weight_requant_ue8m0 = lambda *_args, **_kwargs: False
    model_loader.parameter_mapper = parameter_mapper
    model_loader.utils = model_loader_utils
    sglang_srt.model_loader = model_loader

    sglang_utils = _stub_module(monkeypatch, "sglang.srt.utils", package=True)
    sglang_utils.MultiprocessingSerializer = object
    patch_torch = _stub_module(monkeypatch, "sglang.srt.utils.patch_torch")
    patch_torch.monkey_patch_torch_reductions = lambda: None
    sglang_utils.patch_torch = patch_torch
    sglang_srt.utils = sglang_utils

    weight_sync = _stub_module(monkeypatch, "sglang.srt.weight_sync", package=True)
    tensor_bucket = _stub_module(monkeypatch, "sglang.srt.weight_sync.tensor_bucket")
    tensor_bucket.FlattenedTensorBucket = object
    weight_sync.tensor_bucket = tensor_bucket
    sglang_srt.weight_sync = weight_sync


@pytest.fixture
def transfer_utils_module(monkeypatch: pytest.MonkeyPatch):
    saved_module, saved_attribute = _remove_module(TRANSFER_UTILS_MODULE_NAME)
    _install_import_stubs(monkeypatch)
    module = importlib.import_module(TRANSFER_UTILS_MODULE_NAME)
    try:
        yield module
    finally:
        _restore_module(TRANSFER_UTILS_MODULE_NAME, saved_module, saved_attribute)


@pytest.fixture
def p2p_module(transfer_utils_module):
    saved_module, saved_attribute = _remove_module(P2P_MODULE_NAME)
    module = importlib.import_module(P2P_MODULE_NAME)
    try:
        yield module
    finally:
        _restore_module(P2P_MODULE_NAME, saved_module, saved_attribute)


def _failed_future() -> Future:
    future = Future()
    future.set_exception(RuntimeError("synthetic transfer failure"))
    return future


def _successful_future() -> Future:
    future = Future()
    future.set_result(None)
    return future


class _BlockingFutureList(list[Future[None]]):
    def __init__(
        self,
        futures: list[Future[None]],
        iteration_blocked: Event,
        allow_iteration_finish: Event,
    ) -> None:
        super().__init__(futures)
        self._iteration_blocked = iteration_blocked
        self._allow_iteration_finish = allow_iteration_finish

    def __iter__(self):
        yield from self[:]
        self._iteration_blocked.set()
        if not self._allow_iteration_finish.wait(timeout=1):
            raise TimeoutError("test did not release transfer-future iteration")


def _make_updater(p2p_module, transfer_manager, *, is_source: bool):
    updater = object.__new__(p2p_module.UpdateWeightP2P)
    updater.args = SimpleNamespace()
    updater.weight_version = 0
    updater.is_lora = False
    updater._lora_base_synced = False
    updater._group_name = "test-p2p"
    updater.transfer_plan = SimpleNamespace(
        _gathered_dp_rank=0 if is_source else 1,
        _rollout_num_gpus=1,
    )
    updater.transfer_manager = transfer_manager
    updater._tensor_update_pending = {}
    updater._staged_tensors = {}
    updater._pause_and_prepare_engines = MagicMock()
    updater._gather_and_update_non_expert_weights = MagicMock()
    updater._finalize_and_resume_engines = MagicMock()
    return updater


def _patch_update_lifecycle(monkeypatch: pytest.MonkeyPatch, p2p_module):
    mixin_module = importlib.import_module(
        "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin"
    )
    gloo_group = object()
    progress_bar = object()
    expert_gather_calls = []

    def gather_expert_weights(updater, update_bucket_weight_func, pbar=None):
        expert_gather_calls.append((updater, update_bucket_weight_func, pbar))

    barrier = MagicMock()
    monkeypatch.setattr(
        p2p_module.DistBucketedWeightUpdateMixin,
        "_gather_and_update_expert_weights",
        gather_expert_weights,
    )
    monkeypatch.setattr(mixin_module.dist, "barrier", barrier)
    monkeypatch.setattr(mixin_module, "get_gloo_group", lambda: gloo_group)
    monkeypatch.setattr(mixin_module, "timer", lambda _: nullcontext())
    monkeypatch.setattr(mixin_module, "tqdm", MagicMock(return_value=progress_bar))
    monkeypatch.setattr(p2p_module, "get_gloo_group", lambda: gloo_group)

    return gloo_group, progress_bar, expert_gather_calls, barrier


def test_wait_transfers_raises_for_completed_failure_and_stops_tracking_it(
    caplog: pytest.LogCaptureFixture,
    transfer_utils_module,
) -> None:
    manager = transfer_utils_module.P2PTransferManager(num_workers=1, transfer_timeout=1)
    failed_future = _failed_future()
    manager.transfer_futures.append(failed_future)

    with pytest.raises(RuntimeError) as error:
        manager.wait_transfers()

    assert str(error.value) == TRANSFER_WAIT_FAILURE_MESSAGE
    assert type(error.value.__cause__) is RuntimeError
    assert str(error.value.__cause__) == "synthetic transfer failure"
    assert error.value.__cause__ is failed_future.exception()
    assert manager.failure_cause is error.value.__cause__
    assert manager.transfer_futures == []
    assert failed_future.done() is True
    assert caplog.records[-1].getMessage() == "[P2P] P2P transfer task 0 failed"
    assert caplog.records[-1].exc_info is not None
    assert caplog.records[-1].exc_info[1] is error.value.__cause__


def test_wait_transfers_raises_for_timeout_and_retains_unresolved_future(transfer_utils_module) -> None:
    manager = transfer_utils_module.P2PTransferManager(num_workers=1, transfer_timeout=0)
    unresolved_future = Future()
    manager.transfer_futures.append(unresolved_future)

    with pytest.raises(RuntimeError) as error:
        manager.wait_transfers()

    assert str(error.value) == TRANSFER_WAIT_FAILURE_MESSAGE
    assert type(error.value.__cause__) is TimeoutError
    assert manager.failed is True
    assert manager.failure_cause is error.value.__cause__
    assert manager.transfer_futures == [unresolved_future]
    assert unresolved_future.done() is False


def test_wait_transfers_stops_after_first_failure_and_retains_unresolved_work(transfer_utils_module) -> None:
    manager = transfer_utils_module.P2PTransferManager(num_workers=1, transfer_timeout=1)
    failed_future = _failed_future()
    unresolved_future = MagicMock(spec=Future)
    unresolved_future.done.return_value = False
    manager.transfer_futures.extend([failed_future, unresolved_future])

    with pytest.raises(RuntimeError) as error:
        manager.wait_transfers()

    assert str(error.value) == TRANSFER_WAIT_FAILURE_MESSAGE
    assert error.value.__cause__ is failed_future.exception()
    assert manager.transfer_futures == [unresolved_future]
    assert unresolved_future.result.call_args_list == []


def test_wait_transfer_batch_preserves_concurrent_submission(transfer_utils_module) -> None:
    manager = transfer_utils_module.P2PTransferManager(num_workers=1, transfer_timeout=1)
    completed_future = _successful_future()
    iteration_blocked = Event()
    allow_iteration_finish = Event()
    manager.transfer_futures = _BlockingFutureList(
        [completed_future],
        iteration_blocked,
        allow_iteration_finish,
    )
    wait_errors = []
    submitted_futures = []
    submission_started = Event()

    def wait_for_batch() -> None:
        try:
            manager.wait_transfer_batch([completed_future])
        except Exception as error:
            wait_errors.append(error)

    def submit_transfer() -> None:
        submission_started.set()
        submitted_futures.append(manager.submit_returning_future(lambda: None))

    wait_thread = Thread(target=wait_for_batch)
    submit_thread = Thread(target=submit_transfer)
    wait_thread.start()
    assert iteration_blocked.wait(timeout=1) is True
    assert manager._failure_lock.locked() is True
    submit_thread.start()

    try:
        assert submission_started.wait(timeout=1) is True
        assert manager.transfer_futures == [completed_future]
    finally:
        allow_iteration_finish.set()
        wait_thread.join(timeout=1)
        submit_thread.join(timeout=1)
        manager.executor.shutdown()

    assert wait_thread.is_alive() is False
    assert submit_thread.is_alive() is False
    assert wait_errors == []
    assert manager.transfer_futures == submitted_futures


def test_wait_transfer_batch_applies_timeout_to_each_operation(transfer_utils_module) -> None:
    manager = transfer_utils_module.P2PTransferManager(num_workers=1, transfer_timeout=1)
    first_future = MagicMock(spec=Future)
    second_future = MagicMock(spec=Future)
    first_future.done.return_value = True
    second_future.done.return_value = True
    manager.transfer_futures.extend([first_future, second_future])

    manager.wait_transfer_batch([first_future, second_future])

    assert manager.transfer_futures == []
    assert first_future.result.call_args_list == [call(timeout=1)]
    assert second_future.result.call_args_list == [call(timeout=1)]


def test_failed_non_last_transfer_stops_before_loading_next_replica(
    p2p_module,
    transfer_utils_module,
) -> None:
    manager = transfer_utils_module.P2PTransferManager(num_workers=1, transfer_timeout=1)
    failed_future = _failed_future()
    manager.submit_returning_future = MagicMock(return_value=failed_future)
    manager.submit = MagicMock()
    updater = _make_updater(p2p_module, manager, is_source=True)
    first_replica = MagicMock()
    next_replica = MagicMock()
    first_session = object()
    next_session = object()
    ready_tensors = [("weight", object())]
    transfer_names = ["weight"]
    raw_weight = object()
    observed_converted_tensors = []
    updater._transfer_engine_meta_list = [
        (first_replica, [first_session]),
        (next_replica, [next_session]),
    ]

    def get_transfer_ready_params(converted_tensors):
        observed_converted_tensors.append(list(converted_tensors))
        return transfer_names, ready_tensors

    updater._get_transfer_ready_params = MagicMock(side_effect=get_transfer_ready_params)
    updater._do_p2p_write_one_session = MagicMock()
    converted_tensors = [("raw_weight", raw_weight)]

    updater._update_weight_implementation(converted_tensors)

    assert converted_tensors == []
    assert manager.failed is True
    assert observed_converted_tensors == [[("raw_weight", raw_weight)]]
    assert first_replica.load_weights.call_args_list == [call(ready_tensors)]
    assert next_replica.load_weights.call_args_list == []
    assert manager.submit_returning_future.call_args_list == [
        call(updater._do_p2p_write_one_session, first_session, transfer_names)
    ]
    assert manager.submit.call_args_list == []


def test_timed_out_non_last_transfer_stops_before_loading_next_replica(
    p2p_module,
    transfer_utils_module,
) -> None:
    manager = transfer_utils_module.P2PTransferManager(num_workers=1, transfer_timeout=0)
    unresolved_future = Future()
    manager.submit_returning_future = MagicMock(return_value=unresolved_future)
    manager.submit = MagicMock()
    updater = _make_updater(p2p_module, manager, is_source=True)
    first_replica = MagicMock()
    next_replica = MagicMock()
    first_session = object()
    next_session = object()
    ready_tensors = [("weight", object())]
    transfer_names = ["weight"]
    updater._transfer_engine_meta_list = [
        (first_replica, [first_session]),
        (next_replica, [next_session]),
    ]
    updater._get_transfer_ready_params = MagicMock(return_value=(transfer_names, ready_tensors))
    updater._do_p2p_write_one_session = MagicMock()
    converted_tensors = [("raw_weight", object())]

    updater._update_weight_implementation(converted_tensors)

    assert converted_tensors == []
    assert manager.failed is True
    assert manager.transfer_futures == []
    assert first_replica.load_weights.call_args_list == [call(ready_tensors)]
    assert next_replica.load_weights.call_args_list == []
    assert manager.submit_returning_future.call_args_list == [
        call(updater._do_p2p_write_one_session, first_session, transfer_names)
    ]
    assert manager.submit.call_args_list == []


def test_poisoned_updater_discards_tensors_without_reusing_shared_buffer(
    p2p_module,
    transfer_utils_module,
) -> None:
    manager = transfer_utils_module.P2PTransferManager(num_workers=1, transfer_timeout=1)
    manager.record_failure("synthetic transfer failure")
    updater = _make_updater(p2p_module, manager, is_source=True)
    model_replica = MagicMock()
    updater._transfer_engine_meta_list = [(model_replica, [object()])]
    updater._get_transfer_ready_params = MagicMock()
    updater._do_p2p_write_one_session = MagicMock()
    manager.submit_returning_future = MagicMock()
    manager.submit = MagicMock()
    converted_tensors = [("raw_weight", object())]

    updater._update_weight_implementation(converted_tensors)

    assert converted_tensors == []
    assert updater._get_transfer_ready_params.call_args_list == []
    assert model_replica.load_weights.call_args_list == []
    assert manager.submit_returning_future.call_args_list == []
    assert manager.submit.call_args_list == []


def test_background_failure_stops_later_bucket_before_shared_buffer_reuse(
    p2p_module,
    transfer_utils_module,
) -> None:
    manager = transfer_utils_module.P2PTransferManager(num_workers=1, transfer_timeout=1)
    updater = _make_updater(p2p_module, manager, is_source=True)
    model_replica = MagicMock()
    remote_session = object()
    first_ready_tensors = [("first_weight", object())]
    first_transfer_names = ["first_weight"]
    first_raw_weight = object()
    observed_converted_tensors = []
    updater._transfer_engine_meta_list = [(model_replica, [remote_session])]

    def get_transfer_ready_params(converted_tensors):
        observed_converted_tensors.append(list(converted_tensors))
        return first_transfer_names, first_ready_tensors

    updater._get_transfer_ready_params = MagicMock(side_effect=get_transfer_ready_params)
    updater._do_p2p_write_one_session = MagicMock(side_effect=RuntimeError("synthetic background failure"))
    first_converted_tensors = [("first_raw_weight", first_raw_weight)]

    try:
        updater._update_weight_implementation(first_converted_tensors)
        background_future = manager.transfer_futures[0]

        with pytest.raises(RuntimeError, match="synthetic background failure"):
            background_future.result(timeout=1)

        second_converted_tensors = [("second_raw_weight", object())]
        updater._update_weight_implementation(second_converted_tensors)
    finally:
        manager.executor.shutdown()

    assert first_converted_tensors == []
    assert second_converted_tensors == []
    assert manager.failed is True
    assert type(manager.failure_cause) is RuntimeError
    assert str(manager.failure_cause) == "synthetic background failure"
    assert observed_converted_tensors == [[("first_raw_weight", first_raw_weight)]]
    assert model_replica.load_weights.call_args_list == [call(first_ready_tensors)]
    assert updater._do_p2p_write_one_session.call_args_list == [call(remote_session, first_transfer_names)]
    assert manager.transfer_futures == [background_future]


def test_successful_non_last_transfer_advances_to_next_replica(
    p2p_module,
    transfer_utils_module,
) -> None:
    manager = transfer_utils_module.P2PTransferManager(num_workers=1, transfer_timeout=1)
    successful_future = _successful_future()
    manager.submit_returning_future = MagicMock(return_value=successful_future)
    manager.submit = MagicMock()
    updater = _make_updater(p2p_module, manager, is_source=True)
    first_replica = MagicMock()
    next_replica = MagicMock()
    first_session = object()
    next_session = object()
    ready_tensors = [("weight", object())]
    transfer_names = ["weight"]
    updater._transfer_engine_meta_list = [
        (first_replica, [first_session]),
        (next_replica, [next_session]),
    ]
    updater._get_transfer_ready_params = MagicMock(return_value=(transfer_names, ready_tensors))
    updater._do_p2p_write_one_session = MagicMock()
    converted_tensors = [("raw_weight", object())]

    updater._update_weight_implementation(converted_tensors)

    assert converted_tensors == []
    assert manager.failed is False
    assert first_replica.load_weights.call_args_list == [call(ready_tensors)]
    assert next_replica.load_weights.call_args_list == [call(ready_tensors)]
    assert manager.submit_returning_future.call_args_list == [
        call(updater._do_p2p_write_one_session, first_session, transfer_names)
    ]
    assert manager.submit.call_args_list == [call(updater._do_p2p_write_one_session, next_session, transfer_names)]


def test_local_transfer_failure_reaches_consensus_before_all_ranks_raise(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    p2p_module,
    transfer_utils_module,
) -> None:
    manager = transfer_utils_module.P2PTransferManager(num_workers=1, transfer_timeout=1)
    manager.transfer_futures.append(_failed_future())
    updater = _make_updater(p2p_module, manager, is_source=True)
    updater._tensor_update_pending = {"weight": 1}
    gloo_group, progress_bar, expert_gather_calls, barrier = _patch_update_lifecycle(monkeypatch, p2p_module)
    consensus = MagicMock(return_value=False)
    monkeypatch.setattr(p2p_module, "collective_bool_and", consensus)

    with pytest.raises(RuntimeError) as error:
        updater.update_weights()

    assert str(error.value) == WEIGHT_UPDATE_FAILURE_MESSAGE
    assert consensus.call_args_list == [call(value=False, group=gloo_group)]
    assert updater._pause_and_prepare_engines.call_args_list == [call()]
    assert updater._gather_and_update_non_expert_weights.call_args_list == [
        call(updater._update_weight_implementation, progress_bar)
    ]
    assert expert_gather_calls == [(updater, updater._update_weight_implementation, progress_bar)]
    assert barrier.call_args_list == [call(group=gloo_group), call(group=gloo_group)]
    assert updater._finalize_and_resume_engines.call_args_list == []
    assert [record.getMessage() for record in caplog.records] == ["[P2P] P2P transfer task 0 failed"]


def test_remote_transfer_failure_stops_healthy_rank_before_finalization(
    monkeypatch: pytest.MonkeyPatch,
    p2p_module,
    transfer_utils_module,
) -> None:
    manager = transfer_utils_module.P2PTransferManager(num_workers=1, transfer_timeout=1)
    updater = _make_updater(p2p_module, manager, is_source=False)
    gloo_group, _, expert_gather_calls, barrier = _patch_update_lifecycle(monkeypatch, p2p_module)
    consensus = MagicMock(return_value=False)
    monkeypatch.setattr(p2p_module, "collective_bool_and", consensus)

    with pytest.raises(RuntimeError) as error:
        updater.update_weights()

    assert str(error.value) == WEIGHT_UPDATE_FAILURE_MESSAGE
    assert consensus.call_args_list == [call(value=True, group=gloo_group)]
    assert manager.failed is True
    assert updater._pause_and_prepare_engines.call_args_list == [call()]
    assert updater._gather_and_update_non_expert_weights.call_args_list == [
        call(updater._update_weight_implementation, None)
    ]
    assert expert_gather_calls == [(updater, updater._update_weight_implementation, None)]
    assert barrier.call_args_list == [call(group=gloo_group), call(group=gloo_group)]
    assert updater._finalize_and_resume_engines.call_args_list == []

    del updater._pause_and_prepare_engines
    base_pause = MagicMock()
    monkeypatch.setattr(p2p_module.DistBucketedWeightUpdateMixin, "_pause_and_prepare_engines", base_pause)
    shared_params = {"weight": object()}
    original_engines = [object()]
    original_lock = object()
    updater._shared_params_dict = shared_params
    updater.rollout_engines = original_engines
    updater.rollout_engine_lock = original_lock
    updater._connection_stale = True

    pause_result = updater._pause_and_prepare_engines()

    assert pause_result is None
    assert base_pause.call_args_list == []
    assert updater._shared_params_dict is shared_params

    with pytest.raises(RuntimeError) as reconnect_error:
        updater.connect_rollout_engines([object()], object())

    assert str(reconnect_error.value) == TRAINER_RECREATE_MESSAGE
    assert reconnect_error.value.__cause__ is None
    assert updater._shared_params_dict is shared_params
    assert updater.rollout_engines is original_engines
    assert updater.rollout_engine_lock is original_lock
    assert updater._connection_stale is True


def test_successful_transfers_finalize_and_resume(
    monkeypatch: pytest.MonkeyPatch,
    p2p_module,
    transfer_utils_module,
) -> None:
    manager = transfer_utils_module.P2PTransferManager(num_workers=1, transfer_timeout=1)
    manager.transfer_futures.append(_successful_future())
    updater = _make_updater(p2p_module, manager, is_source=True)
    gloo_group, progress_bar, expert_gather_calls, barrier = _patch_update_lifecycle(monkeypatch, p2p_module)
    consensus = MagicMock(return_value=True)
    monkeypatch.setattr(p2p_module, "collective_bool_and", consensus)

    result = updater.update_weights()

    assert result is None
    assert updater.weight_version == 1
    assert manager.transfer_futures == []
    assert consensus.call_args_list == [call(value=True, group=gloo_group)]
    assert updater._pause_and_prepare_engines.call_args_list == [call()]
    assert updater._gather_and_update_non_expert_weights.call_args_list == [
        call(updater._update_weight_implementation, progress_bar)
    ]
    assert expert_gather_calls == [(updater, updater._update_weight_implementation, progress_bar)]
    assert updater._finalize_and_resume_engines.call_args_list == [call()]
    assert barrier.call_args_list == [
        call(group=gloo_group),
        call(group=gloo_group),
        call(group=gloo_group),
        call(group=gloo_group),
    ]
