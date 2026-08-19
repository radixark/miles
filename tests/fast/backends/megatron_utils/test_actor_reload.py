import importlib
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pytest

from miles.utils.init_once import InitOnce

_ACTOR_MODULE_NAME = "miles.backends.megatron_utils.actor"


@pytest.fixture(scope="module")
def actor_module():
    """Import the megatron actor with the one native dependency the fast environment lacks stubbed out."""
    package = importlib.import_module("miles.backends.megatron_utils")
    missing = object()
    saved_module = sys.modules.get(_ACTOR_MODULE_NAME, missing)
    saved_saver = sys.modules.get("torch_memory_saver", missing)
    saved_package_attr = getattr(package, "actor", missing)

    saver_module = ModuleType("torch_memory_saver")
    saver_module.torch_memory_saver = Mock()
    sys.modules["torch_memory_saver"] = saver_module
    sys.modules.pop(_ACTOR_MODULE_NAME, None)
    if saved_package_attr is not missing:
        delattr(package, "actor")

    try:
        try:
            module = importlib.import_module(_ACTOR_MODULE_NAME)
        except ImportError as e:
            pytest.skip(f"the megatron actor cannot be imported here: {e!r}")
        yield module
    finally:
        sys.modules.pop(_ACTOR_MODULE_NAME, None)
        if saved_module is not missing:
            sys.modules[_ACTOR_MODULE_NAME] = saved_module
        if saved_package_attr is missing:
            if hasattr(package, "actor"):
                delattr(package, "actor")
        else:
            package.actor = saved_package_attr
        if saved_saver is missing:
            sys.modules.pop("torch_memory_saver", None)
        else:
            sys.modules["torch_memory_saver"] = saved_saver


def _args(tmp_path: Path, **overrides) -> Namespace:
    defaults = dict(
        load=str(tmp_path / "pretrain"),
        requested_load=str(tmp_path / "pretrain"),
        save=str(tmp_path / "run"),
        ckpt_step=None,
        ref_ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        lora_rank=0,
        lora_adapter_path=None,
        multi_lora=False,
        colocate=False,
        rematerialize_param_from_master_weight=False,
        non_persistent_ckpt_type=None,
        fp16=False,
        use_precision_aware_optimizer=False,
        optimizer_cpu_offload=False,
        offload_optimizer_states=False,
        debug_rollout_only=False,
        async_save=False,
        offload_train=False,
        stream_optimizer_state_to_disk=False,
        keep_old_actor=False,
        use_pytorch_profiler=False,
        record_memory_history=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _write_checkpoint(directory: Path, *, iteration: int) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest_checkpointed_iteration.txt").write_text(f"{iteration}\n")
    return str(directory)


def _inited_guard() -> InitOnce:
    guard = InitOnce("MegatronTrainRayActor")
    with guard.guarding():
        pass
    return guard


def _actor(actor_module, *, role: str, args: Namespace):
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.role = role
    actor.args = args
    actor._init_once = _inited_guard()
    actor._asleep = False
    actor.with_ref = False
    actor.with_opd_teacher = False
    actor.model = None
    actor.optimizer = None
    actor.opt_param_scheduler = None
    actor._last_rollout_id = None
    return actor


def _watch_load(actor_module, monkeypatch, *, args: Namespace, iteration: int) -> dict[str, Any]:
    model_module = importlib.import_module("miles.backends.megatron_utils.model")
    seen: dict[str, Any] = {}

    def fake_load_checkpoint(*_args: Any, **_kwargs: Any) -> tuple[int, int]:
        seen["args_during_load"] = vars(args).copy()
        return iteration, 0

    monkeypatch.setattr(model_module, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(model_module, "clear_memory", lambda *a, **k: None)
    monkeypatch.setattr(model_module, "check_peak_gpu_memory_after_load", lambda *a, **k: None)
    monkeypatch.setattr(model_module, "check_model_hashes", lambda *a, **k: None)
    monkeypatch.setattr(actor_module, "clear_memory", lambda *a, **k: None)
    return seen


class TestTheCheckpointAReloadRollsBackTo:
    def test_a_reload_reads_the_directory_the_run_was_told_to_load_from(self, actor_module, tmp_path, monkeypatch):
        """The data source reads its own state out of that same directory, so the two have to be the same one."""
        load = _write_checkpoint(tmp_path / "pretrain", iteration=50)
        args = _args(tmp_path)
        seen = _watch_load(actor_module, monkeypatch, args=args, iteration=50)

        assert _actor(actor_module, role="actor", args=args).load_state() == 51
        assert seen["args_during_load"]["load"] == load

    def test_a_reload_reads_what_the_run_asked_for_rather_than_what_a_parse_fell_back_to(
        self, actor_module, tmp_path, monkeypatch
    ):
        """A cold-started parse rewrites `--load` to the reference weights, which a reload must not read."""
        load = _write_checkpoint(tmp_path / "pretrain", iteration=50)
        args = _args(tmp_path, load=str(tmp_path / "reference"), finetune=True)
        seen = _watch_load(actor_module, monkeypatch, args=args, iteration=50)

        _actor(actor_module, role="actor", args=args).load_state()

        assert seen["args_during_load"]["load"] == load

    def test_a_reload_forgets_the_rollout_the_previous_script_was_last_on(self, actor_module, tmp_path, monkeypatch):
        """A freshly started trainer has not trained yet, and a reloaded one has to look the same to the next one."""
        _write_checkpoint(tmp_path / "pretrain", iteration=50)
        args = _args(tmp_path)
        _watch_load(actor_module, monkeypatch, args=args, iteration=50)
        actor = _actor(actor_module, role="actor", args=args)
        actor._last_rollout_id = 49

        actor.load_state()

        assert actor._last_rollout_id is None

    def test_a_reload_clears_the_flags_a_cold_started_parse_left_behind(self, actor_module, tmp_path, monkeypatch):
        """A parse that found no checkpoint set these, and a reload onto a real checkpoint has to load all of it."""
        _write_checkpoint(tmp_path / "pretrain", iteration=50)
        args = _args(tmp_path, finetune=True, no_load_optim=True, no_load_rng=True, ckpt_step=3)
        seen = _watch_load(actor_module, monkeypatch, args=args, iteration=50)

        _actor(actor_module, role="actor", args=args).load_state()

        during = seen["args_during_load"]
        assert (during["finetune"], during["no_load_optim"], during["no_load_rng"], during["ckpt_step"]) == (
            False,
            False,
            False,
            None,
        )

    def test_a_cold_started_parse_does_not_make_a_real_resume_start_over(self, actor_module, tmp_path, monkeypatch):
        """The rollout to resume at is worked out under the overridden arguments, not the restored ones."""
        _write_checkpoint(tmp_path / "pretrain", iteration=50)
        args = _args(tmp_path, finetune=True, no_load_optim=True, no_load_rng=True)
        _watch_load(actor_module, monkeypatch, args=args, iteration=50)

        assert _actor(actor_module, role="actor", args=args).load_state() == 51

    def test_a_reload_leaves_the_arguments_as_it_found_them(self, actor_module, tmp_path, monkeypatch):
        """The override says where this one load reads from; the run's own arguments have to survive it."""
        _write_checkpoint(tmp_path / "pretrain", iteration=50)
        args = _args(tmp_path, finetune=True, no_load_optim=True, no_load_rng=True, ckpt_step=3)
        _watch_load(actor_module, monkeypatch, args=args, iteration=50)

        _actor(actor_module, role="actor", args=args).load_state()

        assert args.load == str(tmp_path / "pretrain")
        assert (args.finetune, args.no_load_optim, args.no_load_rng, args.ckpt_step) == (True, True, True, 3)

    def test_a_critic_reload_reads_the_critic_directory(self, actor_module, tmp_path, monkeypatch):
        """A critic's own arguments carry its checkpoint dirs, so reading them reads the critic's."""
        critic_load = _write_checkpoint(tmp_path / "pretrain_critic", iteration=60)
        args = _args(tmp_path, requested_load=critic_load)
        seen = _watch_load(actor_module, monkeypatch, args=args, iteration=60)

        assert _actor(actor_module, role="critic", args=args).load_state() == 61
        assert seen["args_during_load"]["load"] == critic_load

    def test_a_reload_that_would_cold_start_is_refused(self, actor_module, tmp_path, monkeypatch):
        """The trainer is alive at the rollout the run reached, so a run that cold started is refused here."""
        args = _args(tmp_path, requested_load=str(tmp_path / "run"))
        _watch_load(actor_module, monkeypatch, args=args, iteration=0)

        with pytest.raises(AssertionError):
            _actor(actor_module, role="actor", args=args).load_state()


class TestWhatAReloadRefuses:
    @pytest.mark.parametrize(
        "overrides",
        [
            dict(debug_rollout_only=True),
            dict(lora_rank=8),
            dict(multi_lora=True),
            dict(colocate=True),
            dict(rematerialize_param_from_master_weight=True),
            dict(non_persistent_ckpt_type="local"),
            dict(offload_train=True),
            dict(use_pytorch_profiler=True),
            dict(record_memory_history=True),
            dict(keep_old_actor=True),
        ],
    )
    def test_a_trainer_it_cannot_restore_is_refused_before_anything_is_read(self, actor_module, tmp_path, overrides):
        """Each of these means a checkpoint load cannot put the trainer back where the run really is."""
        actor = _actor(actor_module, role="actor", args=_args(tmp_path, **overrides))

        with pytest.raises(AssertionError):
            actor.load_state()

    def test_a_run_holding_a_second_copy_of_the_actor_says_why_it_is_refused(self, actor_module, tmp_path):
        """`--keep-old-actor` is the one refusal whose reason is not obvious from reading the flag's name."""
        _write_checkpoint(tmp_path / "pretrain", iteration=50)
        actor = _actor(actor_module, role="actor", args=_args(tmp_path, keep_old_actor=True))

        with pytest.raises(AssertionError, match="second copy of the actor"):
            actor.load_state()

    def test_a_run_that_was_never_given_a_load_directory_is_refused(self, actor_module, tmp_path):
        """There is no checkpoint for this reload to restore, and starting the trainer over would replay the run."""
        actor = _actor(actor_module, role="actor", args=_args(tmp_path, requested_load=None))

        with pytest.raises(AssertionError, match="a hot restart needs --load"):
            actor.load_state()
