import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.utils.tracking_utils.ci_history import RECORD_DIR_ENV, CiHistoryBackend


def _recorded(backend: CiHistoryBackend) -> dict[str, list[list[object]]]:
    lines = Path(backend._record_path).read_text().splitlines()
    return {json.loads(line)["metric"]: json.loads(line)["series"] for line in lines}


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CiHistoryBackend:
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    collecting = CiHistoryBackend()
    collecting.init(Namespace())
    return collecting


class TestThePolicyAMetricWasLoggedUnder:
    def test_a_policy_prefixed_metric_is_recorded_under_the_key_it_was_logged_with(self, backend):
        """The whitelist is matched by suffix, and the record then stored it under the bare key."""
        backend.log({"solver/train/grad_norm": 1.5}, step=0)

        assert list(_recorded(backend)) == ["solver/train/grad_norm"]

    def test_two_policies_of_one_run_stay_two_series(self, backend):
        """One series holding step 0 twice is what made the gate's selector refuse the duplicate step."""
        backend.log({"solver/train/grad_norm": 1.5}, step=0)
        backend.log({"verifier/train/grad_norm": 2.5}, step=0)

        assert _recorded(backend) == {
            "solver/train/grad_norm": [[0, 1.5]],
            "verifier/train/grad_norm": [[0, 2.5]],
        }

    def test_a_single_policy_run_is_recorded_exactly_as_it_was(self, backend):
        """An unprefixed key is the key a gate spec of an ordinary run names, and it may not move."""
        backend.log({"train/grad_norm": 1.5}, step=3)

        assert _recorded(backend) == {"train/grad_norm": [[3, 1.5]]}

    def test_every_step_of_one_policy_lands_in_that_policy_series(self, backend):
        """A gate reads a series per key, and steps of one policy scattered over two keys read as gaps."""
        backend.log({"solver/train/grad_norm": 1.0}, step=0)
        backend.log({"solver/train/grad_norm": 2.0}, step=1)

        assert _recorded(backend) == {"solver/train/grad_norm": [[0, 1.0], [1, 2.0]]}

    def test_a_key_outside_the_whitelist_is_still_never_recorded(self, backend):
        """Keeping the logged key must not turn the whitelist into a prefix-matching pass-through."""
        backend.log({"solver/train/something_else": 1.0}, step=0)

        assert not Path(backend._record_path).exists()

    def test_a_non_numeric_value_is_still_refused(self, backend):
        """An authoring error has to fail loudly rather than be dropped from the record."""
        with pytest.raises(TypeError, match="train/grad_norm"):
            backend.log({"solver/train/grad_norm": "high"}, step=0)


class TestCiHistoryBackend:
    def test_a_namespaced_policy_metric_is_captured_under_its_canonical_ci_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A namespaced target must be persisted under its canonical CI metric identity."""
        monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
        backend = CiHistoryBackend()
        backend.init(SimpleNamespace())

        backend.log(metrics={"alpha/train/grad_norm": 2.5}, step=7)

        [record_path] = tmp_path.glob("*.jsonl")
        assert json.loads(record_path.read_text()) == {
            "metric": "alpha/train/grad_norm",
            "series": [[7, 2.5]],
        }
