from __future__ import annotations

import json
import shlex
import subprocess
import sys
import types

import pytest

from miles.backends.sglang_utils import sglang_engine
from miles.backends.sglang_utils.sglang_engine import compute_diffusion_engine_launch_cmd


def _launch_argv(**overrides) -> list[str]:
    kwargs = dict(
        interpreter_prefix=["/usr/bin/python3"],
        base_gpu_id=2,
        sglang_overrides={"model_path": "/models/Z-Image-Turbo"},
        host="10.0.0.5",
        port=30000,
        master_port=9001,
    )
    return shlex.split(compute_diffusion_engine_launch_cmd(**(kwargs | overrides)))


class TestLaunchCommand:
    def test_the_engine_starts_through_the_gpu_pinning_entry_point(self):
        argv = _launch_argv()
        assert argv[:6] == [
            "/usr/bin/python3",
            "-m",
            "miles.backends.sglang_utils.sglang_engine",
            "--base-gpu-id",
            "2",
            "--server-args",
        ]
        assert json.loads(argv[6]) == {
            "model_path": "/models/Z-Image-Turbo",
            "trust_remote_code": True,
            "host": "10.0.0.5",
            "port": 30000,
            "master_port": 9001,
            "num_gpus": 1,
            "warmup_mode": "off",
            "performance_mode": "manual",
        }

    def test_the_group_overrides_win_over_miles_defaults(self):
        overrides = {"model_path": "/m", "performance_mode": "speed", "batching_max_size": 4}
        server_args = json.loads(_launch_argv(sglang_overrides=overrides)[6])
        assert server_args["performance_mode"] == "speed"
        assert server_args["batching_max_size"] == 4

    def test_an_ipv6_host_is_passed_without_brackets(self):
        assert json.loads(_launch_argv(host="[fd00::1]")[6])["host"] == "fd00::1"


class TestEntryPoint:
    @pytest.mark.parametrize(
        ("visible", "base_gpu_id", "expected"), [(None, 3, "3"), ("4,5,6", 1, "5"), ("GPU-a,GPU-b", 0, "GPU-a")]
    )
    def test_the_engine_sees_only_its_own_gpu(self, monkeypatch, visible, base_gpu_id, expected):
        # Set before deleting, so monkeypatch's undo also removes the value pin_gpu writes.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible or "")
        if visible is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        sglang_engine.pin_gpu(base_gpu_id)
        assert sglang_engine.os.environ["CUDA_VISIBLE_DEVICES"] == expected

    def test_the_entry_point_module_loads_no_sglang_srt_before_the_pin(self):
        """sglang.srt initializes CUDA on every GPU the process sees, which the pin must come before."""
        code = "import sys, miles.backends.sglang_utils.sglang_engine; print([m for m in sys.modules if m.startswith('sglang.srt')])"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        assert result.stdout.strip() == "[]", result.stderr

    def test_only_sglang_option_names_pass_before_the_model_loads(self):
        """Registered names pass, the selected pipeline's own ones included; typos, aliases and text-server args fail."""
        server_args = {
            "model_path": "/m",
            "dit_precision": "bf16",
            "dit_config.torch_compile_mode": "max-autotune",
            "kv_cache_quant": "off",
            "warmup_sampling_params": {"width": 512, "height": 512},
            "enable_metrics": True,
            "pipeline_class_name": "Flux3ImagePipeline",
            "flux3_alpha": 6.5,
            "enable_memory_saver": False,
            "master-port": 32001,
            "typo": {},
            "dit_config.no_such_field": 1,
        }
        result = _run_entry_point(server_args)
        assert result.returncode != 0
        assert (
            "sglang.multimodal_gen has no server option named: "
            "['dit_config.no_such_field', 'enable_memory_saver', 'master-port', 'typo']"
        ) in result.stderr


class TestAllocatedArgs:
    def test_a_port_sglang_moved_is_refused_before_launch(self):
        """The broker binds port + 1, which Miles reserved beside the port it allocated, not beside a moved one."""
        allocated = {"num_gpus": 1, "port": 20004, "master_port": 20006}
        resolved = types.SimpleNamespace(num_gpus=1, port=20004, master_port=20043)
        with pytest.raises(ValueError, match=r"resolved \{'master_port': 20043\}"):
            sglang_engine.reject_moved_allocations(allocated, resolved=resolved)
        sglang_engine.reject_moved_allocations(allocated, resolved=types.SimpleNamespace(**allocated))


def _run_entry_point(server_args: dict) -> subprocess.CompletedProcess:
    # A fresh interpreter: torchtitan's compat shim, loaded by this test session, breaks sglang.multimodal_gen imports.
    return subprocess.run(
        [sys.executable, "-m", "miles.backends.sglang_utils.sglang_engine", "--base-gpu-id", "0"]
        + ["--server-args", json.dumps(server_args)],
        capture_output=True,
        text=True,
        timeout=300,
    )
