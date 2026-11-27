"""TOP FP8 masters are installed from --hf-checkpoint by the model provider.

An HF --load is skipped only while the model still holds those weights (an optimizer step or a Megatron load
replaces them), and never ahead of --ckpt-step.
"""

from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

model_module = pytest.importorskip("miles.backends.megatron_utils.model")


def _patch_load_side_effects(stack: ExitStack) -> None:
    for name in (
        "clear_memory",
        "check_peak_gpu_memory_after_load",
        "check_model_hashes",
        "is_first_replica_megatron_main_rank",
    ):
        stack.enter_context(patch.object(model_module, name))


class TestTopFp8InstalledHfCheckpoint:
    @pytest.fixture(autouse=True)
    def _fresh_weight_events(self, monkeypatch):
        from miles.backends.megatron_utils import weight_events

        monkeypatch.setattr(weight_events, "_INSTALLED_HF_CHECKPOINT", None)
        monkeypatch.setattr(weight_events, "_HOLDS_INSTALLED_HF", False)
        return weight_events

    def _load(self, load_dir: Path, *, ckpt_step: int | None = None, finetune: bool = True):
        with ExitStack() as stack:
            loader = stack.enter_context(patch.object(model_module, "load_checkpoint", return_value=(7, True)))
            _patch_load_side_effects(stack)
            output = model_module.load_model_state(
                Namespace(
                    lora_rank=0,
                    megatron_to_hf_mode="core",
                    lora_adapter_path=None,
                    multi_lora_n_adapters=0,
                    multi_lora=False,
                    start_rollout_id=None,
                    backend=Namespace(
                        use_checkpoint_opt_param_scheduler=True,
                        global_batch_size=8,
                        finetune=finetune,
                        load=str(load_dir),
                        pretrained_checkpoint=None,
                        ckpt_step=ckpt_step,
                        exit_on_missing_checkpoint=False,
                        no_load_optim=False,
                    ),
                ),
                model=[object()],
                optimizer=None,
                opt_param_scheduler=None,
                role="actor",
                checkpointing_context=None,
                resume_from_ckpt=True,
            )
        return output, loader

    def _hf(self, tmp_path: Path) -> Path:
        hf = tmp_path / "hf"
        hf.mkdir()
        (hf / "config.json").write_text("{}")
        return hf

    def test_the_installed_checkpoint_is_not_reloaded_while_the_model_holds_it(self, tmp_path, _fresh_weight_events):
        hf = self._hf(tmp_path)
        _fresh_weight_events.mark_hf_checkpoint_installed(str(hf))

        output, loader = self._load(hf)

        loader.assert_not_called()
        assert output.start_rollout_id == 0
        assert _fresh_weight_events.holds_installed_hf_checkpoint()

    @pytest.mark.parametrize("replaced_by", ["optimizer-step", "megatron-load"])
    def test_a_restart_onto_the_installed_checkpoint_after_it_was_replaced_is_refused(
        self, tmp_path, _fresh_weight_events, replaced_by
    ):
        """A hot restart before the first save would otherwise keep the trained weights and reset the optimizer."""
        hf = self._hf(tmp_path)
        _fresh_weight_events.mark_hf_checkpoint_installed(str(hf))
        if replaced_by == "megatron-load":
            run = tmp_path / "run"
            run.mkdir()
            (run / "latest_checkpointed_iteration.txt").write_text("7")
            _, loader = self._load(run, finetune=False)
            loader.assert_called_once()
        else:
            _fresh_weight_events.masters_left_installed_hf()

        assert not _fresh_weight_events.holds_installed_hf_checkpoint()
        with pytest.raises(NotImplementedError, match=r"\[top\].*resume from a Megatron checkpoint"):
            self._load(hf)

    def test_an_explicit_checkpoint_step_is_loaded_rather_than_skipped(self, tmp_path, _fresh_weight_events):
        """``--ckpt-step N`` reads ``iter_N`` even without a tracker, as it does natively."""
        hf = self._hf(tmp_path)
        _fresh_weight_events.mark_hf_checkpoint_installed(str(hf))
        run = tmp_path / "run"
        (run / "iter_0000007").mkdir(parents=True)

        output, loader = self._load(run, ckpt_step=7, finetune=False)

        loader.assert_called_once()
        assert output.loaded_rollout_id == 7
        assert not _fresh_weight_events.holds_installed_hf_checkpoint()

    def test_a_trackerless_non_hf_load_goes_to_the_loader(self, tmp_path, _fresh_weight_events):
        """Only an HF checkpoint can be the install; any other --load is read as it is natively."""
        hf = self._hf(tmp_path)
        _fresh_weight_events.mark_hf_checkpoint_installed(str(hf))
        run = tmp_path / "run"
        (run / "iter_0000007").mkdir(parents=True)

        _, loader = self._load(run, finetune=False)

        loader.assert_called_once()
