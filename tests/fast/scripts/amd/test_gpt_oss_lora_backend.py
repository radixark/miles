import shlex
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.amd import run_gpt_oss_20b_lora as script


class TestGptOssLoraBackend:
    def test_downloads_and_training_use_the_backend_owned_by_the_script_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AMD preparation and training route through the configured backend without losing model arguments."""
        cpu_commands: list[str] = []
        datasets: list[tuple[str, str]] = []
        launches: list[dict[str, Any]] = []
        backend = SimpleNamespace(
            exec_command_cpu=cpu_commands.append,
            hf_download_dataset=lambda repo, *, data_dir: datasets.append((repo, data_dir)),
            execute_train=lambda **kwargs: launches.append(kwargs),
        )
        args = script.ScriptArgs(
            hardware="MI350X",
            num_gpus_per_node=4,
            model_dir="/models",
            data_dir="/datasets",
            megatron_path="/megatron",
            enable_wandb=False,
            extra_args="--seed 17",
        )

        def create_backend(owner: script.ScriptArgs) -> SimpleNamespace:
            assert owner is args
            return backend

        monkeypatch.setattr(script.ScriptArgs, "create_backend", create_backend)
        for name in (
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
            "NCCL_NVLS_ENABLE",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
        ):
            monkeypatch.setenv(name, "0")

        script._download_inputs(args)
        script._execute(args)

        assert cpu_commands == [
            "mkdir -p /models /datasets",
            "hf download lmsys/gpt-oss-20b-bf16 --local-dir /models/gpt-oss-20b-bf16",
        ]
        assert datasets == [("zhuzilin/dapo-math-17k", "/datasets")]
        assert len(launches) == 1
        launch = launches[0]
        assert launch["num_gpus_per_node"] == 4
        assert launch["megatron_model_type"] == "gpt-oss-20b"
        assert launch["megatron_path"] == "/megatron"
        assert "config" not in launch
        tokens = shlex.split(launch["train_args"])
        assert tokens[tokens.index("--hf-checkpoint") + 1] == "/models/gpt-oss-20b-bf16"
        assert tokens[tokens.index("--actor-num-gpus-per-node") + 1] == "4"
        assert tokens[tokens.index("--seed") + 1] == "17"
