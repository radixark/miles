"""LoRA detection, adapter parameters, and training checkpoint state."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

import miles.backends.megatron_utils.lora.utils as lora_utils
from miles.backends.megatron_utils.lora.native.modules.linear import NativeLoRAAdapter
from miles.backends.megatron_utils.lora.utils import adapter_named_parameters, is_lora_enabled, save_lora_checkpoint
from miles.utils.lora.utils import LORA_ADAPTER_NAME, is_lora_weight_name

# ---------------------------------------------------------------------------
# is_lora_enabled
# ---------------------------------------------------------------------------


class TestIsLoraEnabled:
    def test_enabled_by_rank(self):
        args = Namespace(lora_rank=32, lora_adapter_path=None)
        assert is_lora_enabled(args) is True

    def test_enabled_by_adapter_path(self):
        args = Namespace(lora_rank=0, lora_adapter_path="/some/path")
        assert is_lora_enabled(args) is True

    def test_enabled_by_both(self):
        args = Namespace(lora_rank=16, lora_adapter_path="/some/path")
        assert is_lora_enabled(args) is True

    def test_disabled(self):
        args = Namespace(lora_rank=0, lora_adapter_path=None)
        assert is_lora_enabled(args) is False

    def test_disabled_missing_attrs(self):
        args = Namespace()
        assert is_lora_enabled(args) is False


# ---------------------------------------------------------------------------
# is_lora_weight_name / adapter_named_parameters
# ---------------------------------------------------------------------------


class TestIsLoraWeightName:
    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.layers.5.mlp.gate_proj.lora_A.default.weight",
            "base_model.model.layers.5.mlp.gate_proj.lora_B.default.weight",
        ],
    )
    def test_positive(self, name):
        assert is_lora_weight_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ],
    )
    def test_negative(self, name):
        assert is_lora_weight_name(name) is False


class _NativeAdapter(NativeLoRAAdapter):
    def __init__(self):
        super().__init__("model.layers.0.self_attn.q_proj.", (), tp_rank=0)
        self.lora_A = torch.nn.Parameter(torch.zeros(1, 2))
        self.lora_B = torch.nn.Parameter(torch.zeros(2, 1))

    def export_plan(self, gather):
        return []


class TestAdapterNamedParameters:
    """Adapter parameters are whatever the adapter modules own; their names are irrelevant."""

    def test_native_adapter_parameters_are_found_whatever_they_are_called(self):
        model = torch.nn.Module()
        model.base = torch.nn.Linear(2, 2)
        model.anything = _NativeAdapter()
        model.anything.register_parameter("magnitude", torch.nn.Parameter(torch.ones(2)))

        assert sorted(adapter_named_parameters([model])[0]) == [
            "anything.lora_A",
            "anything.lora_B",
            "anything.magnitude",
        ]

    def test_bridge_wrappers_contribute_everything_but_the_wrapped_base(self):
        AdapterWrapper = pytest.importorskip("megatron.bridge.peft.adapter_wrapper").AdapterWrapper

        adapter = torch.nn.Module()
        adapter.linear_in = torch.nn.Linear(2, 1, bias=False)
        model = torch.nn.Module()
        model.linear_qkv = AdapterWrapper(torch.nn.Linear(2, 2), adapter)

        assert sorted(adapter_named_parameters([model])[0]) == ["linear_qkv.adapter.linear_in.weight"]

    def test_base_parameters_named_like_adapters_are_not_adapters(self):
        model = torch.nn.Module()
        model.mm_adapter = torch.nn.Linear(2, 2)
        model.register_parameter("lora_scale", torch.nn.Parameter(torch.ones(1)))

        assert adapter_named_parameters([model]) == [{}]


# ---------------------------------------------------------------------------
# LORA_ADAPTER_NAME constant
# ---------------------------------------------------------------------------


def test_lora_adapter_name_constant():
    assert LORA_ADAPTER_NAME == "miles_lora"


class _AdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.adapter = _NativeAdapter()


def _single_rank(monkeypatch):
    monkeypatch.setattr(
        lora_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(rank=0), pp=SimpleNamespace(rank=0)),
    )


class TestSaveLoraCheckpointTrainingState:
    def _save(self, tmp_path, *, no_save_optim, scheduler=None):
        publisher = SimpleNamespace(write_adapter=lambda *_: None)

        model = [_AdapterModel()]
        args = Namespace(megatron_to_hf_mode="bridge", no_save_optim=no_save_optim)
        optimizer = SimpleNamespace(state_dict=lambda: {"step": 7})
        save_lora_checkpoint(
            model,
            args,
            str(tmp_path / "checkpoint"),
            publisher=publisher,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            iteration=3,
        )
        return sorted(path.name for path in (tmp_path / "checkpoint").iterdir())

    @staticmethod
    def _state(tmp_path):
        return torch.load(tmp_path / "checkpoint" / "training_state_rank0.pt", weights_only=False)

    def test_training_state_is_written_by_default(self, tmp_path):
        scheduler = SimpleNamespace(state_dict=lambda: {"lr": 0.5})
        files = self._save(tmp_path, no_save_optim=False, scheduler=scheduler)

        assert files == ["adapter_megatron_rank0.pt", "training_state_rank0.pt"]
        state = self._state(tmp_path)
        assert state["optimizer"] == {"step": 7}
        assert state["opt_param_scheduler"] == {"lr": 0.5}
        assert state["iteration"] == 3

    def test_no_save_optim_drops_the_optimizer_and_keeps_the_resume_metadata(self, tmp_path):
        """--no-save-optim is about optimizer state; losing the step and the LR schedule with it
        would silently restart a resumed run from iteration 0."""
        scheduler = SimpleNamespace(state_dict=lambda: {"lr": 0.5})
        files = self._save(tmp_path, no_save_optim=True, scheduler=scheduler)

        assert files == ["adapter_megatron_rank0.pt", "training_state_rank0.pt"]
        state = self._state(tmp_path)
        assert state["optimizer"] is None
        assert state["opt_param_scheduler"] == {"lr": 0.5}
        assert state["iteration"] == 3


class TestDistributedOptimizerParameterState:
    """DistributedOptimizer.state_dict() omits the fp32 masters and Adam moments; loading it alone
    leaves uninitialized moments and the initial masters, which the first step() writes back."""

    @staticmethod
    def _distributed(calls):
        bucket = [{"param": torch.ones(2), "exp_avg": torch.zeros(2), "gbuf_local_start": 0, "gbuf_local_end": 2}]
        return SimpleNamespace(
            get_parameter_state_dp_reshardable=lambda: {"per_bucket_numel": {}, 0: {torch.float32: [bucket]}},
            load_parameter_state_from_dp_reshardable=lambda state: calls.append(("parameter_state", state)),
        )

    @staticmethod
    def _optimizer(calls):
        return SimpleNamespace(
            state_dict=lambda: {"step": 7},
            load_state_dict=lambda state: calls.append(("load_state_dict", state)),
            reload_model_params=lambda: calls.append(("reload_model_params", None)),
        )

    def test_save_writes_each_distributed_optimizers_parameter_state(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(lora_utils, "_distributed_optimizers", lambda _optimizer: [self._distributed(calls)])
        args = Namespace(no_save_optim=False, megatron_to_hf_mode="bridge")
        publisher = SimpleNamespace(write_adapter=lambda *_args: None)

        save_lora_checkpoint(
            [_AdapterModel()], args, str(tmp_path / "ckpt"), publisher=publisher, optimizer=self._optimizer(calls)
        )

        state = torch.load(tmp_path / "ckpt" / "training_state_rank0.pt", weights_only=False)
        (parameter_state,) = state["parameter_state"]
        torch.testing.assert_close(parameter_state[0][torch.float32][0][0]["param"], torch.ones(2))


class TestRestoreLoraAdapter:
    """--lora-adapter-path restores exactly what its format carries; anything the run asks for and
    the checkpoint cannot give fails instead of silently training a fresh adapter or optimizer."""

    @staticmethod
    def _args(tmp_path, *, no_load_optim, mode="raw"):
        return Namespace(lora_adapter_path=str(tmp_path), no_load_optim=no_load_optim, megatron_to_hf_mode=mode)

    @staticmethod
    def _optimizer(calls):
        return SimpleNamespace(
            load_state_dict=lambda state: calls.append(("load_state_dict", state)),
            reload_model_params=lambda: calls.append(("reload_model_params", None)),
        )

    @staticmethod
    def _write_native(tmp_path, **training_state):
        torch.save(
            {"adapter.lora_A": torch.ones(1, 2), "adapter.lora_B": torch.ones(2, 1)},
            tmp_path / "adapter_megatron_rank0.pt",
        )
        if training_state:
            state = {"iteration": 5, "optimizer": {"step": 7}, "opt_param_scheduler": {"lr": 0.5}} | training_state
            torch.save(state, tmp_path / "training_state_rank0.pt")

    def _restore(self, tmp_path, calls, *, no_load_optim, scheduler=None, mode="raw"):
        return lora_utils.restore_lora_adapter(
            [_AdapterModel()],
            self._args(tmp_path, no_load_optim=no_load_optim, mode=mode),
            optimizer=self._optimizer(calls),
            opt_param_scheduler=scheduler,
        )

    def test_virtual_pipeline_chunks_round_trip_their_own_adapters(self, tmp_path, monkeypatch):
        _single_rank(monkeypatch)
        saved = [_AdapterModel(), _AdapterModel()]
        for value, chunk in enumerate(saved, start=1):
            chunk.adapter.lora_A.data.fill_(value)
        publisher = SimpleNamespace(write_adapter=lambda *_: None)
        save_lora_checkpoint(saved, Namespace(megatron_to_hf_mode="bridge"), str(tmp_path), publisher=publisher)
        restored = [_AdapterModel(), _AdapterModel()]

        lora_utils.restore_lora_adapter(
            restored, self._args(tmp_path, no_load_optim=True), optimizer=None, opt_param_scheduler=None
        )

        assert [chunk.adapter.lora_A.flatten().tolist() for chunk in restored] == [[1.0, 1.0], [2.0, 2.0]]

    def test_a_full_checkpoint_restores_the_optimizer_scheduler_and_step(self, tmp_path, monkeypatch):
        _single_rank(monkeypatch)
        calls = []
        parameter_state = {"per_bucket_numel": {}, 0: {torch.float32: [[{"param": torch.ones(2), "padding": False}]]}}
        distributed = SimpleNamespace(
            load_parameter_state_from_dp_reshardable=lambda state: calls.append(("parameter_state", state))
        )
        monkeypatch.setattr(lora_utils, "_distributed_optimizers", lambda _optimizer: [distributed])
        self._write_native(tmp_path, parameter_state=[parameter_state])
        scheduler_loads = []

        iteration = self._restore(
            tmp_path, calls, no_load_optim=False, scheduler=SimpleNamespace(load_state_dict=scheduler_loads.append)
        )

        assert iteration == 5
        assert [name for name, _ in calls] == ["load_state_dict", "parameter_state"]
        assert scheduler_loads == [{"lr": 0.5}]

    def test_no_load_optim_refreshes_the_masters_and_keeps_the_step_and_schedule(self, tmp_path, monkeypatch):
        _single_rank(monkeypatch)
        calls = []
        self._write_native(tmp_path, parameter_state=[])
        scheduler_loads = []

        iteration = self._restore(
            tmp_path, calls, no_load_optim=True, scheduler=SimpleNamespace(load_state_dict=scheduler_loads.append)
        )

        assert iteration == 5
        assert calls == [("reload_model_params", None)]
        assert scheduler_loads == [{"lr": 0.5}]

    @pytest.mark.parametrize(
        "training_state, message",
        [
            ({}, "no optimizer state"),
            ({"optimizer": None, "parameter_state": None}, "no optimizer state"),
            ({"optimizer": {"step": 7}}, "predates"),
        ],
        ids=["no-training-state", "saved-with-no-save-optim", "pre-parameter-state-format"],
    )
    def test_an_unrestorable_optimizer_fails_unless_no_load_optim(
        self, tmp_path, monkeypatch, training_state, message
    ):
        _single_rank(monkeypatch)
        self._write_native(tmp_path, **training_state)

        with pytest.raises(AssertionError, match=message):
            self._restore(tmp_path, [], no_load_optim=False)

    def test_an_hf_adapter_loads_weights_only(self, tmp_path, monkeypatch):
        (tmp_path / "adapter_model.safetensors").write_bytes(b"")
        loads = []
        monkeypatch.setattr(lora_utils, "load_lora_adapter_hf", lambda model, path: loads.append(path))
        _single_rank(monkeypatch)
        calls = []

        assert self._restore(tmp_path, calls, no_load_optim=True) is None
        assert loads == [str(tmp_path.resolve())]
        assert calls == [("reload_model_params", None)]
        with pytest.raises(AssertionError, match="no optimizer state"):
            self._restore(tmp_path, [], no_load_optim=False)

    def test_an_hf_adapter_requires_raw_mode(self, tmp_path, monkeypatch):
        (tmp_path / "adapter_model.safetensors").write_bytes(b"")
        _single_rank(monkeypatch)

        with pytest.raises(AssertionError, match="requires --megatron-to-hf-mode raw"):
            self._restore(tmp_path, [], no_load_optim=True, mode="bridge")

    def test_a_directory_without_an_adapter_fails(self, tmp_path, monkeypatch):
        _single_rank(monkeypatch)

        with pytest.raises(AssertionError, match="holds neither"):
            self._restore(tmp_path, [], no_load_optim=True)

    def test_a_shard_that_does_not_match_the_adapter_fails(self, tmp_path, monkeypatch):
        """A silently partial load resumes training on a half-initialized adapter."""
        _single_rank(monkeypatch)
        torch.save(
            {"adapter.lora_A": torch.ones(1, 2), "stale_lora_C": torch.ones(1)}, tmp_path / "adapter_megatron_rank0.pt"
        )

        with pytest.raises(RuntimeError, match=r"missing=\['adapter.lora_B'\], unexpected=\['stale_lora_C'\]"):
            self._restore(tmp_path, [], no_load_optim=True)

    def test_shards_saved_under_another_layout_fail(self, tmp_path, monkeypatch):
        """Falling through to fresh adapter weights would hide a resharding mistake."""
        _single_rank(monkeypatch)
        torch.save({"adapter.lora_A": torch.ones(1, 2)}, tmp_path / "adapter_megatron_rank1.pt")

        with pytest.raises(AssertionError, match="none for global rank 0"):
            self._restore(tmp_path, [], no_load_optim=True)
