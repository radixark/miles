from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher import test_entrypoint
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import router, trainer

from miles.utils.workers.worker_spec import BaseSpec


def _launch_static_roster(*, monkeypatch: pytest.MonkeyPatch, directory: Path) -> dict[str, Any]:
    selected_specs = [router(), trainer(num_cells=1, gpus_per_cell=1)]
    original = test_entrypoint._stub_launch_inputs

    def stub_inputs(patched: pytest.MonkeyPatch, *, specs: list[BaseSpec], colocate: bool = False) -> None:
        original(patched, specs=selected_specs, colocate=colocate)

    monkeypatch.setattr(test_entrypoint, "_stub_launch_inputs", stub_inputs)
    test_entrypoint._record_launch(monkeypatch, directory, ci_run=False)
    values = list(directory.rglob("values-*.yaml"))
    assert len(values) == 1
    return yaml.safe_load(values[0].read_text())
