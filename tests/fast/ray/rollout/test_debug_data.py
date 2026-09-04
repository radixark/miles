from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tests.fast.ray.rollout.conftest import make_args, make_sample, make_samples_grouped

from miles.ray.rollout.debug_data import RolloutDataInjectionUtil, load_debug_rollout_data, save_debug_rollout_data

# ----------------------------- save / load round-trip -----------------------------


class TestRoundTrip:
    def test_train_path_round_trip(self, tmp_path: Path):
        path_template = str(tmp_path / "rollout_{rollout_id}.pt")
        args_save = make_args(save_debug_rollout_data=path_template)
        args_load = make_args(load_debug_rollout_data=path_template)

        original = make_samples_grouped(2, 3)
        save_debug_rollout_data(args_save, original, rollout_id=7, evaluation=False)
        assert (tmp_path / "rollout_7.pt").exists()

        loaded, metadata = load_debug_rollout_data(args_load, rollout_id=7)
        assert len(loaded) == len(original)
        assert metadata == {}
        for orig, got in zip(original, loaded, strict=True):
            assert got.index == orig.index
            assert got.response_length == orig.response_length

    def test_metadata_round_trip(self, tmp_path: Path):
        """Metadata passed to save comes back verbatim from load."""
        path_template = str(tmp_path / "rollout_{rollout_id}.pt")
        args_save = make_args(save_debug_rollout_data=path_template)
        args_load = make_args(load_debug_rollout_data=path_template)

        original_metadata = {"dynamic_global_batch_size": 24}
        save_debug_rollout_data(
            args_save, make_samples_grouped(1, 2), rollout_id=3, evaluation=False, metadata=original_metadata
        )

        _loaded, metadata = load_debug_rollout_data(args_load, rollout_id=3)
        assert metadata == original_metadata

    def test_load_file_without_metadata_key_returns_empty_metadata(self, tmp_path: Path):
        """Files recorded before metadata support load with metadata defaulting to {}."""
        path = tmp_path / "rollout_5.pt"
        torch.save(dict(rollout_id=5, samples=[make_sample().to_dict()]), path)
        args_load = make_args(load_debug_rollout_data=str(tmp_path / "rollout_{rollout_id}.pt"))

        loaded, metadata = load_debug_rollout_data(args_load, rollout_id=5)
        assert len(loaded) == 1
        assert metadata == {}

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

        loaded, _metadata = load_debug_rollout_data(args_load, rollout_id="eval_4")
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
        loaded, _metadata = load_debug_rollout_data(args, rollout_id=0)
        assert len(loaded) == 10

    def test_ratio_half_takes_first_and_last_chunks(self, saved_path_factory):
        """``data[: rough // 2] + data[-rough // 2 :]`` with rough = int(N * ratio).

        For 10 rows, ratio=0.5: ``rough = 5``, ``rough // 2 = 2``, ``-5 // 2 = -3``
        (Python floor-division on negatives), so the slice yields ``data[:2] +
        data[-3:]`` = 5 items: indices [0, 1, 7, 8, 9]."""
        template = saved_path_factory(10)
        args = make_args(load_debug_rollout_data=template, load_debug_rollout_data_subsample=0.5)
        loaded, _metadata = load_debug_rollout_data(args, rollout_id=0)
        assert len(loaded) == 5
        assert [s.index for s in loaded] == [0, 1, 7, 8, 9]


# ----------------------------- CI rollout-data injection -----------------------------


def _save_recording(tmp_path: Path, rollout_id: int, *, metadata: dict | None = None) -> str:
    """Record one rollout file in --save-debug-rollout-data format; return the template."""
    template = str(tmp_path / "rollout_{rollout_id}.pt")
    args = make_args(save_debug_rollout_data=template)
    save_debug_rollout_data(
        args, [make_sample(index=rollout_id)], rollout_id=rollout_id, evaluation=False, metadata=metadata
    )
    return template


class TestShouldInjectRolloutData:
    def test_false_when_injection_not_configured(self):
        """Without --ci-inject-rollout-data-path, no rollout is injected."""
        args = make_args()
        assert RolloutDataInjectionUtil.should_inject(args, rollout_id=0) is False

    @pytest.mark.parametrize(("rollout_id", "expected"), [(2, False), (3, True), (4, True)])
    def test_injects_only_at_or_after_start_rollout_id(self, rollout_id: int, expected: bool):
        """Rollouts before the start id keep their generated data; later ones are injected."""
        args = make_args(
            ci_inject_rollout_data_path="/recorded/{rollout_id}.pt",
            ci_inject_rollout_data_start_rollout_id=3,
        )
        assert RolloutDataInjectionUtil.should_inject(args, rollout_id=rollout_id) is expected


class TestLoadInjectedRolloutData:
    def test_round_trip_returns_samples_and_metadata(self, tmp_path: Path):
        """Injection loads samples and metadata from a --save-debug-rollout-data recording."""
        template = _save_recording(tmp_path, 3, metadata={"dynamic_global_batch_size": 16})
        args = make_args(ci_inject_rollout_data_path=template, ci_inject_rollout_data_start_rollout_id=3)

        data, metadata = RolloutDataInjectionUtil.load(args, rollout_id=3)

        assert [s.index for s in data] == [3]
        assert metadata == {"dynamic_global_batch_size": 16}

    def test_missing_recording_fails_loudly(self, tmp_path: Path):
        """A missing recorded file raises with the resolved path instead of training on wrong data."""
        template = str(tmp_path / "rollout_{rollout_id}.pt")
        args = make_args(ci_inject_rollout_data_path=template, ci_inject_rollout_data_start_rollout_id=3)

        with pytest.raises(AssertionError, match="rollout_7.pt"):
            RolloutDataInjectionUtil.load(args, rollout_id=7)


def _make_paired_sample(prompt_tokens: list[int], response_tokens: list[int]):
    return make_sample(tokens=prompt_tokens + response_tokens, response_length=len(response_tokens))


class TestAssertInjectedRolloutDataMatchesGenerated:
    _PROMPT: list[int] = [101, 102, 103]

    def test_identical_data_passes(self):
        """Bitwise-identical generated and injected data trivially passes the match check."""
        samples = [_make_paired_sample(self._PROMPT, list(range(20))) for _ in range(4)]
        paired = [_make_paired_sample(self._PROMPT, list(range(20))) for _ in range(4)]

        RolloutDataInjectionUtil.assert_matches_generated(
            make_args(), generated=samples, injected=paired, rollout_id=3
        )

    def test_few_flipped_tokens_pass(self):
        """ulp-drift-level divergence (one flipped token per response) stays above the 0.9 threshold."""
        generated = [_make_paired_sample(self._PROMPT, list(range(20))) for _ in range(4)]
        injected = [_make_paired_sample(self._PROMPT, [999] + list(range(1, 20))) for _ in range(4)]

        RolloutDataInjectionUtil.assert_matches_generated(
            make_args(), generated=generated, injected=injected, rollout_id=3
        )

    def test_threshold_comes_from_args(self):
        """A lower --ci-inject-rollout-data-min-match-ratio accepts what the default rejects."""
        generated = [_make_paired_sample(self._PROMPT, list(range(20)))]
        injected = [_make_paired_sample(self._PROMPT, list(range(10)) + [999] * 10)]

        RolloutDataInjectionUtil.assert_matches_generated(
            make_args(ci_inject_rollout_data_min_match_ratio=0.4),
            generated=generated,
            injected=injected,
            rollout_id=3,
        )

    def test_mostly_diverged_responses_fail(self):
        """Responses matching at only 50% (wrong engine weights) fail the threshold loudly."""
        generated = [_make_paired_sample(self._PROMPT, list(range(20))) for _ in range(4)]
        injected = [_make_paired_sample(self._PROMPT, list(range(10)) + [999] * 10) for _ in range(4)]

        with pytest.raises(AssertionError, match="match the injected recording"):
            RolloutDataInjectionUtil.assert_matches_generated(
                make_args(), generated=generated, injected=injected, rollout_id=3
            )

    def test_length_divergence_counts_as_mismatch(self):
        """A response twice as long as the recording scores 0.5 even if the shared prefix matches."""
        generated = [_make_paired_sample(self._PROMPT, list(range(20)))]
        injected = [_make_paired_sample(self._PROMPT, list(range(10)))]

        with pytest.raises(AssertionError, match="match the injected recording"):
            RolloutDataInjectionUtil.assert_matches_generated(
                make_args(), generated=generated, injected=injected, rollout_id=3
            )

    def test_prompt_mismatch_fails_as_wiring_error(self):
        """Differing prompt tokens mean broken pairing, not drift — hard fail regardless of responses."""
        generated = [_make_paired_sample([101, 102, 103], list(range(20)))]
        injected = [_make_paired_sample([201, 202, 203], list(range(20)))]

        with pytest.raises(AssertionError, match="prompt tokens mismatch"):
            RolloutDataInjectionUtil.assert_matches_generated(
                make_args(), generated=generated, injected=injected, rollout_id=3
            )

    def test_sample_count_mismatch_fails(self):
        """Different sample counts cannot be paired and fail before any token comparison."""
        generated = [_make_paired_sample(self._PROMPT, list(range(20))) for _ in range(3)]
        injected = [_make_paired_sample(self._PROMPT, list(range(20))) for _ in range(2)]

        with pytest.raises(AssertionError, match="sample count mismatch"):
            RolloutDataInjectionUtil.assert_matches_generated(
                make_args(), generated=generated, injected=injected, rollout_id=3
            )
