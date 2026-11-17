from pathlib import Path

import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from transformers import Qwen2Config

from miles.utils.args.runtime import TrainerConfig
from miles.utils.debug_utils.run_megatron.worker import main as worker
from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs
from miles.utils.workers.serving.utils import override_argv


class _ModelLoaded(Exception):
    pass


class TestStandaloneCheckpointLoading:
    def test_reference_loading_is_scoped_to_model_initialization(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Standalone model loading receives runtime fallback fields and restores immutable trainer configuration."""
        reference = tmp_path / "reference"
        hf = tmp_path / "hf"
        Qwen2Config(
            hidden_size=128, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, intermediate_size=256
        ).save_pretrained(hf)
        config = parse_megatron_test_config(
            "--ffn-hidden-size",
            "256",
            "--untie-embeddings-and-output-weights",
            "--norm-epsilon",
            "1e-6",
            "--debug-train-only",
            "--no-offload-train",
            "--load",
            str(tmp_path / "run"),
            "--ref-load",
            str(reference),
            "--hf-checkpoint",
            str(hf),
        )
        initialized: list[TrainerConfig] = []

        def initialize(*, args: TrainerConfig) -> None:
            assert args.backend.load == str(reference)
            assert args.backend.finetune
            initialized.append(args)

        def build(args: TrainerConfig, script: WorkerScriptArgs) -> None:
            assert args is initialized[0]
            assert args.backend.load == str(reference)
            assert args.backend.no_load_optim and args.backend.no_load_rng
            raise _ModelLoaded

        monkeypatch.setattr(worker, "parse_args", lambda: config)
        monkeypatch.setattr(worker, "_initialize_megatron", initialize)
        monkeypatch.setattr(worker.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(worker, "_print_config", lambda *args: None)
        monkeypatch.setattr(worker, "setup_replay_before_model", lambda script: None)
        monkeypatch.setattr(worker, "_build_and_load_model", build)
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")

        with override_argv(
            ["--script-hf-checkpoint", str(hf), "--script-token-ids-file", str(tmp_path / "tokens.json")]
        ):
            with pytest.raises(_ModelLoaded):
                worker.main()

        assert "load" not in vars(initialized[0].backend)
        assert "finetune" not in vars(initialized[0].backend)
