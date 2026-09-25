"""Execute production collection methods without starting Ray or GPU workers."""
import ast
import asyncio
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

ROOT = Path(__file__).resolve().parents[2]


def load_functions(path: Path, names: set[str], scope: dict) -> dict:
    tree = ast.parse(path.read_text())
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    for node in functions:
        node.decorator_list = []
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    return scope


def test_rollout_only_empty_and_nonempty_transport() -> None:
    forbidden = Mock(side_effect=AssertionError("training transformation must not run"))
    scope = dict(asyncio=asyncio, time=time, logger=logging.getLogger(__name__), timer=lambda *a: nullcontext(),
                 assert_weight_version_is_published=Mock(), assert_samples_weight_version_sane=Mock(),
                 dashboard_hooks=SimpleNamespace(report_data_buffer=Mock()),
                 RolloutFnTrainInput=lambda **kw: SimpleNamespace(**kw),
                 call_rollout_function=lambda fn, inp: fn(inp),
                 postprocess_rollout_data=forbidden, convert_samples_to_train_data=forbidden,
                 split_train_data_by_dp=forbidden, save_debug_rollout_data=Mock(), log_rollout_data=Mock(),
                 RolloutDataInjectionUtil=SimpleNamespace(should_inject=lambda *a: False))
    load_functions(ROOT / "miles/ray/rollout/rollout_executor.py", {"get", "_get_rollout_data"}, scope)
    worker = SimpleNamespace(args=SimpleNamespace(debug_rollout_only=True, load_debug_rollout_data=None),
                             data_source=SimpleNamespace(), weight_version=0,
                             _rollouts_since_weight_version_publish=0, use_legacy_rollout_v1=False)
    worker._get_rollout_data = MethodType(scope["_get_rollout_data"], worker)
    for groups, expected in [([[SimpleNamespace(index=1)]], [1]), ([], []), ([[SimpleNamespace(index=3)]], [3])]:
        worker.generate_rollout = lambda inp: SimpleNamespace(samples=groups, metrics={})
        output = asyncio.run(scope["get"](worker, 0))
        assert output == {"sample_indices": expected, "data_ref": []}
    assert scope["save_debug_rollout_data"].call_count == 3
    assert scope["log_rollout_data"].call_count == 2
    forbidden.assert_not_called()


def test_driver_continues_after_empty_wave() -> None:
    args = SimpleNamespace(fully_async=False, colocate_memory_peak_device="cpu", api_server_port=None,
                           check_weight_update_equal=False, offload_rollout=False, num_rollout=3,
                           start_rollout_id=0, eval_interval=None, debug_rollout_only=True,
                           debug_exit_after_rollout=None, use_critic=False)
    executor = SimpleNamespace(get=SimpleNamespace(remote=AsyncMock(return_value={"sample_indices": [], "data_ref": []})),
                               save=SimpleNamespace(remote=AsyncMock()), dispose=SimpleNamespace(remote=AsyncMock()))
    inference = SimpleNamespace(prepare_rollout=AsyncMock(), dispose=AsyncMock())
    actor = SimpleNamespace(train=AsyncMock(side_effect=AssertionError("trainer must not be called")), dispose=AsyncMock())
    dispatcher = SimpleNamespace(drain=AsyncMock())
    scope = dict(asyncio=asyncio, configure_logger=Mock(), MainProcessIdentity=Mock(),
                 maybe_start_periodic_pyspy_dump=Mock(), launch_worker_manager=Mock(),
                 object_store=SimpleNamespace(init_instance=Mock()), init_tracking=Mock(),
                 create_rollout_components=AsyncMock(return_value=(inference, executor, 30)),
                 create_training_models=AsyncMock(return_value=(actor, None)),
                 maybe_start_mini_ft_controller=Mock(), update_weights=AsyncMock(),
                 EvalDispatcher=lambda *a: dispatcher, logger=logging.getLogger(__name__))
    load_functions(ROOT / "train.py", {"train"}, scope)
    asyncio.run(scope["train"](args))
    assert [c.args[0] for c in executor.get.remote.call_args_list] == [0, 1, 2]
    assert executor.save.remote.await_count == 3
    actor.train.assert_not_called()
    actor.dispose.assert_awaited_once()
    dispatcher.drain.assert_awaited_once()
