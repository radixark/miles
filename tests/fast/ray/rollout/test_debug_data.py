from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tests.fast.ray.rollout.conftest import make_args, make_sample, make_samples_grouped

from miles.ray.rollout.debug_data import load_debug_rollout_data, save_debug_rollout_data

# ----------------------------- save / load round-trip -----------------------------


class TestRoundTrip:
    def test_train_path_round_trip(self, tmp_path: Path):
        path_template = str(tmp_path / "rollout_{rollout_id}.pt")
        args_save = make_args(save_debug_rollout_data=path_template)
        args_load = make_args(load_debug_rollout_data=path_template)

        original = make_samples_grouped(2, 3)
        save_debug_rollout_data(args_save, original, rollout_id=7, evaluation=False)
        assert (tmp_path / "rollout_7.pt").exists()

        loaded = load_debug_rollout_data(args_load, rollout_id=7)
        assert len(loaded) == len(original)
        for orig, got in zip(original, loaded, strict=True):
            assert got.index == orig.index
            assert got.response_length == orig.response_length

    def test_evaluation_path_round_trip_flattens_dataset_dict(self, tmp_path: Path):
        """eval-mode save flattens dict-of-{name: {samples: [...]}} into one list.

        The loader doesn't know about eval mode — it just reads the flat list.
        So a round-trip should yield the concatenation of all per-dataset samples.
        Save bakes ``"eval_"`` into the rollout_id segment of the filename, so we
        load by passing the already-prefixed string."""
        path_template = str(tmp_path / "{rollout_id}.pt")
        args_save = make_args(save_debug_rollout_data=path_template)
        args_load = make_args(load_debug_rollout_data=path_template)

        eval_data = {
            "ds_a": {"samples": [make_sample(index=i, response=f"a{i}") for i in range(3)]},
            "ds_b": {"samples": [make_sample(index=10 + i, response=f"b{i}") for i in range(2)]},
        }
        save_debug_rollout_data(args_save, eval_data, rollout_id=4, evaluation=True)
        saved_path = tmp_path / "eval_4.pt"
        assert saved_path.exists()

        loaded = load_debug_rollout_data(args_load, rollout_id="eval_4")
        assert len(loaded) == 5
        assert sorted(s.index for s in loaded) == [0, 1, 2, 10, 11]
        # ds_a samples come first (insertion order is preserved by dict iteration)
        assert [s.response for s in loaded] == ["a0", "a1", "a2", "b0", "b1"]

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        nested = tmp_path / "nested" / "deeper"
        path_template = str(nested / "r_{rollout_id}.pt")
        args = make_args(save_debug_rollout_data=path_template)
        save_debug_rollout_data(args, [make_sample()], rollout_id=0, evaluation=False)
        assert nested.exists() and (nested / "r_0.pt").exists()

    def test_save_does_nothing_when_template_is_none(self, monkeypatch):
        """When ``save_debug_rollout_data`` is unset, the function must short-circuit
        before touching torch.save (so no I/O happens regardless of cwd / data shape)."""
        args = make_args(save_debug_rollout_data=None)
        called = []
        monkeypatch.setattr(torch, "save", lambda *a, **kw: called.append((a, kw)))
        save_debug_rollout_data(args, [make_sample()], rollout_id=0, evaluation=False)
        assert called == []


# ----------------------------- subsample -----------------------------


class TestSubsample:
    @pytest.fixture
    def saved_path_factory(self, tmp_path: Path):
        def _save(n_samples: int) -> str:
            path_template = str(tmp_path / "r_{rollout_id}.pt")
            args = make_args(save_debug_rollout_data=path_template)
            samples = [make_sample(index=i, response="r" + str(i)) for i in range(n_samples)]
            save_debug_rollout_data(args, samples, rollout_id=0, evaluation=False)
            return path_template

        return _save

    def test_ratio_one_returns_full_data(self, saved_path_factory):
        template = saved_path_factory(10)
        args = make_args(load_debug_rollout_data=template, load_debug_rollout_data_subsample=1.0)
        loaded = load_debug_rollout_data(args, rollout_id=0)
        assert len(loaded) == 10

    def test_ratio_half_takes_first_and_last_chunks(self, saved_path_factory):
        """``data[: rough // 2] + data[-rough // 2 :]`` with rough = int(N * ratio).

        For 10 rows, ratio=0.5: ``rough = 5``, ``rough // 2 = 2``, ``-5 // 2 = -3``
        (Python floor-division on negatives), so the slice yields ``data[:2] +
        data[-3:]`` = 5 items: indices [0, 1, 7, 8, 9]."""
        template = saved_path_factory(10)
        args = make_args(load_debug_rollout_data=template, load_debug_rollout_data_subsample=0.5)
        loaded = load_debug_rollout_data(args, rollout_id=0)
        assert len(loaded) == 5
        assert [s.index for s in loaded] == [0, 1, 7, 8, 9]
