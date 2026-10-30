import json

import pytest
from tests.fast.utils.workers.serving.registered_serve import serve_config_argv
from tests.fast.utils.workers.serving.serve_smoke_worker import SmokeServeSpec, SmokeWorkerConfig

from miles.ray.specs.entrypoint import SERVE_SPEC_CLASSES


class _DefaultedWorkerConfig(SmokeWorkerConfig):
    worker_argv: list[str] = []


@pytest.fixture
def incomplete_worker_argv(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setitem(SERVE_SPEC_CLASSES, SmokeServeSpec.worker_type, SmokeServeSpec)
    monkeypatch.setattr(SmokeServeSpec, "config_class", _DefaultedWorkerConfig)
    argv = serve_config_argv(spec_class=SmokeServeSpec, config=_DefaultedWorkerConfig(rpc_port=8000))
    index = argv.index("--config") + 1
    payload = json.loads(argv[index])
    del payload["args"]["worker_argv"]
    argv[index] = json.dumps(payload)
    monkeypatch.setattr(
        SmokeServeSpec, "create", lambda *args, **kwargs: pytest.fail("Created a spec from incomplete config")
    )
    return argv
