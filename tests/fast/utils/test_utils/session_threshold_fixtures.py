import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from scripts.tools.verify_session_tito_tokenizer import _parse_session_verify_args, _with_session_verify_defaults
from transformers import Qwen2Config

from miles.utils.test_utils import session_verify_runner
from miles.utils.workers.serving.utils import override_argv


@dataclass
class SessionBackendBoundary:
    calls: list[dict[str, Any]] = field(default_factory=list)
    metrics_path: Path | None = None

    def execute_train(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        self.metrics_path = Path(kwargs["extra_env_vars"]["MILES_SESSION_VERIFY_METRICS_PATH"])
        records = [
            dict(driver_events=["append_tool"], hard_mismatch_count=1, hard_mismatch_types=["special_token_count"]),
            dict(driver_events=["append_tool"], hard_mismatch_count=0, hard_mismatch_types=[]),
        ]
        self.metrics_path.write_text("".join(json.dumps(record) + "\n" for record in records))


@dataclass
class SessionThresholdInputs:
    args: argparse.Namespace
    backend: SessionBackendBoundary


@pytest.fixture
def session_threshold_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, threshold: float
) -> SessionThresholdInputs:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    Qwen2Config(
        hidden_size=128, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, intermediate_size=256
    ).save_pretrained(model_directory)
    argv = _with_session_verify_defaults(
        [
            "--tito-model",
            "qwen3",
            "--hf-checkpoint",
            str(model_directory),
            "--actor-num-gpus-per-node",
            "1",
            "--rollout-num-gpus-per-engine",
            "1",
            "--rollout-batch-size",
            "2",
            "--session-verify-cycles",
            "2",
            "--tool-call-failure-mode",
            "append_user",
            "--no-enable-sample-ownership-checker",
            "--sglang-reasoning-parser",
            "qwen3",
            "--sglang-tool-call-parser",
            "qwen25",
            "--ci-tito-special-token-count-threshold",
            str(threshold),
        ]
    )
    with override_argv(argv):
        args = _parse_session_verify_args()
    backend = SessionBackendBoundary()
    monkeypatch.setattr(
        session_verify_runner.command_utils, "default_config", lambda: SimpleNamespace(create_backend=lambda: backend)
    )
    monkeypatch.setattr(session_verify_runner, "PROMPT_DATA_PATH", str(tmp_path / "prompt.jsonl"))
    monkeypatch.setattr(session_verify_runner.tempfile, "tempdir", str(tmp_path))
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.setenv(key, "http://proxy.invalid")
    return SessionThresholdInputs(args=args, backend=backend)
