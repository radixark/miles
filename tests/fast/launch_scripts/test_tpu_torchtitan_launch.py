import re
import shlex
from pathlib import Path

import pytest
from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)

_DENSE = "scripts/tpu/run_qwen3_4b_torchtitan.py"
_MOE = "scripts/tpu/run_qwen3_5_torchtitan.py"


def _record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: str, entrypoint: str, **overrides):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    call_entrypoint(import_launch_script(REPO_ROOT / script), entrypoint, overrides, sandbox=tmp_path)
    return recording.commands


def _flag(tokens: list[str], flag: str) -> str:
    return tokens[tokens.index(flag) + 1]


class TestSubHostEngines:
    @pytest.mark.parametrize("engine_tp", [1, 2])
    def test_the_engine_devices_split_into_servers_with_their_own_ranks_and_ports(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine_tp: int
    ) -> None:
        """Server k drives devices 4 + k*tp .., sits at the slice ranks behind the trainer's 4, and takes port
        30000 + 10k with the rendezvous port derived from it, so two servers never draw the same one."""
        (command,) = _record(monkeypatch, tmp_path, _DENSE, "sglang", engine_tp=engine_tp)
        # "source env.sh && (server 0) & (server 1) > sglang-30010.log 2>&1 & ... wait -n; ..."
        servers = re.findall(r"\((SGLANG_TPU_SHARED_SLICE=.*?)\)( > sglang-\d+\.log 2>&1)? & ", command)
        num_engines = 4 // engine_tp

        assert len(servers) == num_engines
        assert command.startswith("source /sgl-workspace/sglang/benchmark/tpu/common_env.sh && (")
        assert command.endswith("wait -n; status=$?; kill $(jobs -p) 2> /dev/null; wait; exit $status")
        for index, (server, log_redirect) in enumerate(servers):
            tokens = shlex.split(server)
            offset = 4 + index * engine_tp
            assert f"SGLANG_TPU_SHARED_SLICE={offset},8,localhost:30108" in tokens
            assert _flag(tokens, "--tp-size") == str(engine_tp)
            assert _flag(tokens, "--base-gpu-id") == str(offset)
            assert _flag(tokens, "--port") == str(30000 + 10 * index)
            assert _flag(tokens, "--nccl-port") == str(30466 + 10 * index)
            # the first server keeps the terminal; the others log next to it
            assert log_redirect == ("" if index == 0 else f" > sglang-{30000 + 10 * index}.log 2>&1")

    @pytest.mark.parametrize("engine_tp", [1, 2])
    def test_the_trainer_drives_every_server_as_an_engine_of_that_size(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine_tp: int
    ) -> None:
        tokens = shlex.split(_record(monkeypatch, tmp_path, _DENSE, "train", engine_tp=engine_tp)[-1])
        num_engines = 4 // engine_tp

        addrs = tokens[tokens.index("--rollout-external-engine-addrs") + 1 :][:num_engines]
        assert addrs == [f"http://localhost:{30000 + 10 * index}" for index in range(num_engines)]
        assert _flag(tokens, "--rollout-num-gpus") == "4"
        assert _flag(tokens, "--rollout-num-gpus-per-engine") == str(engine_tp)
        assert _flag(tokens, "--update-weight-transfer-mode") == "ici"

    def test_one_tp4_server_is_the_default_and_pins_no_rendezvous_port(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The single-server command is what it was (the launch-script snapshots hold the full text)."""
        (command,) = _record(monkeypatch, tmp_path, _DENSE, "sglang")
        tokens = shlex.split(command)

        assert " & " not in command and "--nccl-port" not in tokens
        assert "SGLANG_TPU_SHARED_SLICE=4,8,localhost:30108" in tokens
        assert (_flag(tokens, "--tp-size"), _flag(tokens, "--base-gpu-id"), _flag(tokens, "--port")) == (
            "4",
            "4",
            "30000",
        )
        train = shlex.split(_record(monkeypatch, tmp_path, _DENSE, "train")[-1])
        assert _flag(train, "--rollout-external-engine-addrs") == "http://localhost:30000"
        assert _flag(train, "--rollout-num-gpus-per-engine") == "4"

    @pytest.mark.parametrize("entrypoint", ["sglang", "train"])
    def test_a_size_that_does_not_split_the_engine_devices_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entrypoint: str
    ) -> None:
        with pytest.raises(ValueError, match="--engine-tp 3 must be 1, 2 or 4"):
            _record(monkeypatch, tmp_path, _DENSE, entrypoint, engine_tp=3)

    @pytest.mark.parametrize("entrypoint", ["sglang", "train"])
    def test_a_moe_launcher_keeps_its_experts_over_the_whole_engine(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entrypoint: str
    ) -> None:
        """Its --ep-size / --sglang-ep-size are the engine's 4 devices, which a smaller server cannot serve."""
        with pytest.raises(ValueError, match="takes --engine-tp 4 only"):
            _record(monkeypatch, tmp_path, _MOE, entrypoint, model_name="Qwen3.6-35B-A3B", engine_tp=2)
