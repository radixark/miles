import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.utils.tracking_utils.ci_history import RECORD_DIR_ENV, CiHistoryBackend


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
            "metric": "train/grad_norm",
            "series": [[7, 2.5]],
        }
