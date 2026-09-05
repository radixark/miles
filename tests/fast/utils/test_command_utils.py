import json
import os
import platform
import shlex
import sys
from types import SimpleNamespace

import pytest
from tests.fast.launch_scripts.py_harness import launcher_hardware_literals
from tests.fast.utils.command_recorder import patch_helper, record_commands


import miles.utils.external_utils.command_utils as command_utils
from miles.utils.external_utils.command_utils import base_backend, common
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainRequest
from miles.utils.external_utils.command_utils.ray_backend import command as ray_command
from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend
from miles.utils.external_utils.model_args_utils import load_model_args
from miles.utils.file_arg_utils import resolve_file_arg
from miles.utils.typer_utils import SCRIPT_ENV_VAR_PREFIX


def _backend():
    return command_utils.default_config().create_backend()


@pytest.fixture(autouse=True)
def bare_environment(monkeypatch):
    """Every test here launches onto ray, and a workbench pod exports variables that choose another backend."""
    for name in [key for key in os.environ if key.startswith(SCRIPT_ENV_VAR_PREFIX)]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def commands(monkeypatch):
    recorded = record_commands(monkeypatch)
    patch_helper(monkeypatch, "_check_has_nvlink", lambda self: False, backend_class=RayCommandBackend)
    for name in ("MILES_SCRIPT_EXTERNAL_RAY", "RAY_ADDRESS", "NCCL_NVLS_ENABLE", "WANDB_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    return recorded


def _runtime_env(submit_command):
    arg = next(arg for arg in shlex.split(submit_command) if arg.startswith("--runtime-env-json="))
    return json.loads(arg.split("=", 1)[1])["env_vars"]


class TestExecuteTrainConfig:
    def test_num_nodes_reads_the_slurm_allocation_when_the_config_is_built(self, monkeypatch):
        """A plain class-level default would bake in the allocation at import and ignore later changes."""
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "8")
        assert command_utils.ExecuteTrainConfig().num_nodes == 8

        monkeypatch.delenv("SLURM_JOB_NUM_NODES")
        assert command_utils.ExecuteTrainConfig().num_nodes == 1


class TestConvertCheckpoint:
    def test_preserves_source_paths_on_the_pythonpath(self, monkeypatch, tmp_path):
        """The converter runs out-of-process, so miles and megatron must be on its PYTHONPATH."""
        commands = []
        monkeypatch.setenv("PYTHONPATH", "/sglang:/existing")
        patch_helper(
            monkeypatch,
            "exec_command_gpu",
            lambda self, cmd, capture_output=False, **kwargs: commands.append(cmd),
        )

        _backend().convert_checkpoint(
            model_name="model",
            megatron_model_type="qwen3-4B",
            num_gpus_per_node=1,
            dir_dst=str(tmp_path),
            megatron_path="/megatron",
        )

        expected = os.pathsep.join([str(command_utils.repo_base_dir), "/megatron", "/sglang", "/existing"])
        assert f"PYTHONPATH={shlex.quote(expected)} " in commands[0]

    def test_defaults_the_hf_checkpoint_to_the_model_name(self, commands, tmp_path):
        """Callers that only pass a model name get /root/models/<model_name> as the source."""
        _backend().convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        assert "--hf-checkpoint /root/models/Qwen3-4B " in commands[0]
        assert f"--save {tmp_path}/Qwen3-4B_torch_dist " in commands[0]

    def test_an_explicit_hf_checkpoint_wins_over_the_default(self, commands, tmp_path):
        """Callers converting a checkpoint that does not live under /root/models must be honoured."""
        _backend().convert_checkpoint(
            model_name="Qwen3-4B",
            megatron_model_type="qwen3-4B",
            num_gpus_per_node=8,
            dir_dst=str(tmp_path),
            hf_checkpoint="/elsewhere/Qwen3-4B",
        )

        assert "--hf-checkpoint /elsewhere/Qwen3-4B " in commands[0]
        assert "/root/models" not in commands[0]

    def test_skips_an_already_released_destination(self, commands, tmp_path):
        """A tracker file reading 'release' means the conversion already finished."""
        dst = tmp_path / "Qwen3-4B_torch_dist"
        dst.mkdir()
        (dst / "latest_checkpointed_iteration.txt").write_text("release\n")

        _backend().convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        assert commands == []

    def test_reruns_when_the_tracker_holds_an_iteration(self, commands, tmp_path):
        """Only the literal 'release' marker counts as done; an iteration number does not."""
        dst = tmp_path / "Qwen3-4B_torch_dist"
        dst.mkdir()
        (dst / "latest_checkpointed_iteration.txt").write_text("42")

        _backend().convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        assert len(commands) == 1

    def test_multinode_uses_torchrun_rendezvous_placeholders(self, commands, tmp_path):
        """Multi-node conversion must template the placeholders exec_command_multi_node substitutes."""
        _backend().convert_checkpoint(
            model_name="Qwen3-4B",
            megatron_model_type="qwen3-4B",
            num_gpus_per_node=8,
            multinode=True,
            num_nodes=2,
            dir_dst=str(tmp_path),
            extra_args="--extra 1",
        )

        assert "--master-addr {{master_addr}}" in commands[0]
        assert "--nnodes={{nnodes}}" in commands[0]
        assert "--node-rank {{node_rank}}" in commands[0]
        assert commands[0].endswith("--extra 1")

    def test_single_node_omits_the_rendezvous_placeholders(self, commands, tmp_path):
        """A single-node conversion has nothing to rendezvous with."""
        _backend().convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        assert "--master-addr" not in commands[0]


class TestRsyncCmd:
    def test_creates_the_destination_before_copying(self):
        """rsync fails on a missing destination, so the mkdir has to precede it."""
        assert command_utils.rsync_cmd("/src", "/dst") == "mkdir -p /dst && rsync -a --info=progress2 /src/ /dst"

    def test_copies_the_contents_rather_than_nesting_the_source(self):
        """Without the trailing slash rsync would land the tree at /dst/src and every reader would miss it."""
        assert command_utils.rsync_cmd("/src", "/dst").endswith(" /src/ /dst")


class TestHfDownloadDataset:
    def test_strips_the_namespace_from_the_local_dir(self, commands):
        """The local directory is named after the dataset, not after owner/dataset."""
        _backend().hf_download_dataset("zhuzilin/dapo-math-17k", data_dir="/data")

        assert commands == ["hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /data/dapo-math-17k"]


class TestFp8CastBf16:
    def test_skips_when_the_output_index_already_exists(self, commands, tmp_path):
        """A safetensors index in the destination means the cast already ran."""
        (tmp_path / "model.safetensors.index.json").write_text("{}")

        _backend().fp8_cast_bf16("/src", str(tmp_path))

        assert commands == []

    def test_runs_when_the_output_is_absent(self, commands, tmp_path):
        """Without the index file the cast must actually be invoked."""
        _backend().fp8_cast_bf16("/src", str(tmp_path))

        assert "--input-fp8-hf-path /src " in commands[0]
        assert f"--output-bf16-hf-path {tmp_path} " in commands[0]


class TestStartMooncakeMaster:
    def test_reuses_a_ready_server(self, monkeypatch):
        """An already listening master must not be restarted out from under its clients."""
        commands = []
        waits = []
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command._is_tcp_server_ready",
            lambda host, port: True,
        )
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command.run_shell_command", commands.append
        )
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command.wait_for_server_ready",
            lambda *args, **kwargs: waits.append((args, kwargs)),
        )

        ray_command.start_mooncake_master()

        assert commands == []
        assert waits == []

    def test_lets_the_os_choose_the_metrics_port(self, monkeypatch):
        """Binding port zero atomically avoids collisions with other listeners."""
        commands = []
        waits = []
        monkeypatch.setattr(command_utils, "_is_tcp_server_ready", lambda host, port: False)
        monkeypatch.setattr(command_utils, "exec_command_cpu", commands.append)
        monkeypatch.setattr(
            command_utils, "wait_for_server_ready", lambda *args, **kwargs: waits.append((args, kwargs))
        )

        command_utils.start_mooncake_master()

        assert len(commands) == 1
        assert "mooncake_master --rpc_port 50051 --metrics_port 0" in commands[0]
        assert waits == [(("127.0.0.1", 50051), {"timeout": 30})]

    def test_restarts_and_waits_until_ready(self, monkeypatch, tmp_path):
        """A dead master is replaced and the caller blocks until the new one answers."""
        commands = []
        waits = []
        log_path = tmp_path / "mooncake master.log"
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command._is_tcp_server_ready",
            lambda host, port: False,
        )
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command.run_shell_command", commands.append
        )
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command.wait_for_server_ready",
            lambda *args, **kwargs: waits.append((args, kwargs)),
        )

        ray_command.start_mooncake_master(rpc_port=50151, metrics_port=50152, timeout=12, log_path=log_path)

        assert len(commands) == 1
        assert "pkill -x mooncake_master" in commands[0]
        assert "mooncake_master --rpc_port 50151 --metrics_port 50152" in commands[0]
        assert f"> {shlex.quote(str(log_path))} 2>&1 &" in commands[0]
        assert waits == [(("127.0.0.1", 50151), {"timeout": 12})]

    def test_reports_the_log_when_startup_fails(self, monkeypatch, tmp_path):
        """The log is the only clue about why the master refused to come up."""
        log_path = tmp_path / "mooncake_master.log"
        log_path.write_text("bind failed\nfatal startup error\n")
        commands = []
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command._is_tcp_server_ready",
            lambda host, port: False,
        )
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command.run_shell_command", commands.append
        )

        def fail_wait(*args, **kwargs):
            raise RuntimeError("not ready")

        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command.wait_for_server_ready", fail_wait
        )

        with pytest.raises(RuntimeError, match="fatal startup error"):
            ray_command.start_mooncake_master(log_path=log_path)

        assert len(commands) == 2
        assert all("pkill -x mooncake_master" in command for command in commands)


class TestPrepareCmd:
    def test_runs_the_command_on_every_node_before_submitting_the_job(self, commands):
        """The trainers read what the command stages, so a job submitted first would read nothing."""
        _backend().execute_train(
            train_args="", num_gpus_per_node=1, megatron_model_type="qwen3-4B", prepare_cmd={"trainer": "cp a b"}
        )

        prepared = next(index for index, command in enumerate(commands) if command.endswith("cp a b"))
        submitted = next(index for index, command in enumerate(commands) if "ray job submit" in command)
        assert commands[prepared].startswith("[multi_node ")
        assert prepared < submitted

    def test_runs_the_command_after_ray_is_up(self, commands):
        """exec_command_multi_node reaches the nodes through ray, which a preceding command cannot use."""
        _backend().execute_train(
            train_args="", num_gpus_per_node=1, megatron_model_type="qwen3-4B", prepare_cmd={"trainer": "cp a b"}
        )

        started = next(index for index, command in enumerate(commands) if "ray start --head" in command)
        prepared = next(index for index, command in enumerate(commands) if command.endswith("cp a b"))
        assert started < prepared

    def test_leaves_the_run_alone_when_nothing_is_asked_for(self, commands):
        """A run that stages nothing must not gain an empty command on every node."""
        _backend().execute_train(train_args="", num_gpus_per_node=1, megatron_model_type="qwen3-4B")

        assert not any(command.startswith("[multi_node ") for command in commands)

    def test_rejects_a_role_no_backend_knows_how_to_prepare(self, commands):
        """A misspelled role would be silently dropped and the run would read unstaged data."""
        with pytest.raises(AssertionError, match="rollout"):
            _backend().execute_train(
                train_args="",
                num_gpus_per_node=1,
                megatron_model_type="qwen3-4B",
                prepare_cmd={"rollout": "cp a b"},
            )


class TestExecuteTrain:
    def test_exports_unbuffered_python_to_ray(self, monkeypatch):
        """Ray start and job submit must export the correctly spelled PYTHONUNBUFFERED."""
        commands = []
        monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY", raising=False)
        monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
        patch_helper(monkeypatch, "exec_command_cpu", lambda self, cmd, capture_output=False: commands.append(cmd))
        patch_helper(monkeypatch, "_check_has_nvlink", lambda self: False, backend_class=RayCommandBackend)

        _backend().execute_train(
            train_args="",
            num_gpus_per_node=1,
            megatron_model_type="qwen3-4B",
        )

        exports = [command for command in commands if "export PYTHONUNBUFFERED" in command]
        assert len(exports) == 2
        assert not any("PYTHONBUFFERED" in command for command in commands)
        assert all("export PYTHONUNBUFFERED=1 &&" in command for command in exports)

    def test_unbuffers_the_ray_workers_too(self, monkeypatch):
        """An export only reaches the submitting client; the ray workers read the runtime environment."""
        commands = []
        monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
        monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
        patch_helper(monkeypatch, "exec_command_cpu", lambda self, cmd, capture_output=False: commands.append(cmd))
        patch_helper(monkeypatch, "_check_has_nvlink", lambda self: False, backend_class=RayCommandBackend)

        _backend().execute_train(train_args="", num_gpus_per_node=1, megatron_model_type="qwen3-4B")

        runtime_env_arg = next(arg for arg in shlex.split(_submit(commands)) if arg.startswith("--runtime-env-json="))
        assert json.loads(runtime_env_arg.split("=", 1)[1])["env_vars"]["PYTHONUNBUFFERED"] == "1"

    def test_preserves_source_paths_in_the_ray_runtime(self, monkeypatch):
        """A caller-supplied PYTHONPATH must be prepended to, not replace, the checkouts miles needs."""
        commands = []
        monkeypatch.setenv("PYTHONPATH", "/sglang:/existing")
        monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
        monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
        patch_helper(monkeypatch, "exec_command_cpu", lambda self, cmd, capture_output=False: commands.append(cmd))
        patch_helper(monkeypatch, "_check_has_nvlink", lambda self: False, backend_class=RayCommandBackend)

        _backend().execute_train(
            train_args="",
            num_gpus_per_node=1,
            megatron_model_type="qwen3-4B",
            megatron_path="/megatron",
            extra_env_vars={"PYTHONPATH": "/custom:/sglang", "QUOTED_VALUE": "it's preserved"},
        )

        submit_command = commands[-1]
        runtime_env_arg = next(arg for arg in shlex.split(submit_command) if arg.startswith("--runtime-env-json="))
        runtime_env = json.loads(runtime_env_arg.split("=", 1)[1])
        expected = os.pathsep.join([str(command_utils.repo_base_dir), "/megatron", "/custom", "/sglang", "/existing"])
        assert runtime_env["env_vars"]["PYTHONPATH"] == expected
        assert runtime_env["env_vars"]["QUOTED_VALUE"] == "it's preserved"

    def test_rejects_fsdp_with_a_megatron_model_type(self, commands):
        """FSDP runs have no megatron model config, so a model type means the launcher is confused."""
        with pytest.raises(AssertionError):
            _backend().execute_train(
                train_args="--train-backend fsdp", num_gpus_per_node=8, megatron_model_type="qwen"
            )

    def test_rejects_megatron_without_a_model_type(self, commands):
        """Without a model type the submitted job would carry no architecture flags at all."""
        with pytest.raises(AssertionError):
            _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type=None)

    def test_starts_a_local_ray_cluster_by_default(self, commands):
        """Without MILES_SCRIPT_EXTERNAL_RAY the launcher owns the ray cluster lifecycle."""
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert "ray stop --force; " in commands[0]
        assert "ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats" in commands[1]

    def test_leaves_an_external_ray_cluster_alone(self, commands, monkeypatch):
        """With an external cluster we must neither stop nor start ray."""
        monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")

        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert not any("ray stop" in command or "ray start" in command for command in commands)
        assert not any("pkill -9 ray" in command for command in commands)

    def test_runs_the_callback_before_submitting(self, commands):
        """before_ray_job_submit exists to prepare state the job will read."""
        _backend().execute_train(
            train_args="",
            num_gpus_per_node=8,
            megatron_model_type="qwen3-4B",
            before_ray_job_submit=lambda: commands.append("CALLBACK"),
        )

        assert commands.index("CALLBACK") < len(commands) - 1
        assert "ray start --head" in commands[commands.index("CALLBACK") - 1]
        assert "ray job submit" in commands[-1]

    def test_can_skip_the_ray_job_submit(self, commands, monkeypatch):
        """Preparation-only runs disable the submit but still clean up and start ray."""
        monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "0")

        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert not any("ray job submit" in command for command in commands)

    def test_expands_the_model_config_into_the_submitted_command(self, commands):
        """The megatron model type is expanded into the argv its model script declares."""
        _backend().execute_train(train_args="--x 1", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        submit = _submit(commands)
        assert "--num-layers 36 " in submit
        assert "source" not in submit
        assert submit.endswith("--x 1 --deploy-component all")

    def test_quotes_the_model_args_the_shell_would_otherwise_reinterpret(self, commands):
        """--moe-layer-freq [1,1,1] is a glob; an unquoted token expands against the launch directory."""
        _backend().execute_train(train_args="--x 1", num_gpus_per_node=8, megatron_model_type="deepseek-v3-5layer")

        submit = _submit(commands)
        assert "--moe-layer-freq '[0,0,0,1,1]'" in submit
        assert shlex.split(submit)[shlex.split(submit).index("--moe-layer-freq") + 1] == "[0,0,0,1,1]"

    def test_model_args_survive_a_shell_round_trip_unchanged(self, commands):
        """Quoting is only correct if the training process still receives exactly the declared argv."""
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="deepseek-v3-5layer")

        declared = load_model_args("deepseek-v3-5layer").split()
        submitted = shlex.split(_submit(commands))
        model_argv = submitted[: len(submitted) - 2]

        assert submitted[len(submitted) - 2 :] == ["--deploy-component", "all"]
        assert model_argv[len(model_argv) - len(declared) :] == declared

    def test_omits_the_model_args_for_fsdp(self, commands):
        """FSDP has no megatron model config to expand."""
        _backend().execute_train(train_args="--train-backend fsdp", num_gpus_per_node=8, megatron_model_type=None)

        assert "--num-layers" not in _submit(commands)

    def test_drops_cuda_device_max_connections_for_fsdp(self, commands):
        """Pinning it to 1 breaks computation/communication overlap on FSDP."""
        _backend().execute_train(train_args="--train-backend fsdp", num_gpus_per_node=8, megatron_model_type=None)

        assert "CUDA_DEVICE_MAX_CONNECTIONS" not in _runtime_env(_submit(commands))

    def test_pins_cuda_device_max_connections_for_megatron(self, commands):
        """Megatron requires the serialized copy engine ordering."""
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert _runtime_env(_submit(commands))["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"

    def test_derives_nvls_from_nvlink_detection(self, commands, monkeypatch):
        """NCCL_NVLS_ENABLE follows the detected topology when it is not preset."""
        patch_helper(monkeypatch, "_check_has_nvlink", lambda self: True, backend_class=RayCommandBackend)

        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert _runtime_env(_submit(commands))["NCCL_NVLS_ENABLE"] == "1"

    def test_lets_the_environment_override_nvls(self, commands, monkeypatch):
        """An explicit NCCL_NVLS_ENABLE wins over topology detection."""
        patch_helper(monkeypatch, "_check_has_nvlink", lambda self: True, backend_class=RayCommandBackend)
        monkeypatch.setenv("NCCL_NVLS_ENABLE", "0")

        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert _runtime_env(_submit(commands))["NCCL_NVLS_ENABLE"] == "0"

    def test_forwards_selected_nccl_variables_only_when_present(self, commands, monkeypatch):
        """Optional debug knobs are passed through, and absent ones must not appear as empty strings."""
        monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth0")
        monkeypatch.delenv("NCCL_DEBUG", raising=False)

        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        runtime_env = _runtime_env(_submit(commands))
        assert runtime_env["NCCL_SOCKET_IFNAME"] == "eth0"
        assert "NCCL_DEBUG" not in runtime_env

    def test_bypasses_the_proxy_for_the_master_node(self, commands, monkeypatch):
        """Routing intra-cluster traffic through a proxy hangs the job."""
        monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")

        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        runtime_env = _runtime_env(_submit(commands))
        assert runtime_env["no_proxy"] == "127.0.0.1,10.0.0.1"
        assert runtime_env["MASTER_ADDR"] == "10.0.0.1"

    def test_enables_cuda_core_dumps_on_request(self, commands):
        """The core dump knobs only appear when the config asks for them."""
        config = command_utils.ExecuteTrainConfig(cuda_core_dump=True, output_dir="/out")

        config.create_backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        runtime_env = _runtime_env(_submit(commands))
        assert runtime_env["CUDA_ENABLE_COREDUMP_ON_EXCEPTION"] == "1"
        assert runtime_env["CUDA_COREDUMP_FILE"] == "/out/cuda_coredump_%h.%p.%t"

    def test_omits_cuda_core_dumps_by_default(self, commands):
        """Core dumps are expensive, so they must stay off unless asked for."""
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert "CUDA_ENABLE_COREDUMP_ON_EXCEPTION" not in _runtime_env(_submit(commands))

    def test_lets_config_extra_env_vars_win_over_the_argument(self, commands):
        """The CLI-supplied overrides are applied last so an operator can always override a script."""
        config = command_utils.ExecuteTrainConfig(extra_env_vars="MY_VAR=from_config")

        config.create_backend().execute_train(
            train_args="",
            num_gpus_per_node=8,
            megatron_model_type="qwen3-4B",
            extra_env_vars={"MY_VAR": "from_argument", "OTHER": "kept"},
        )

        runtime_env = _runtime_env(_submit(commands))
        assert runtime_env["MY_VAR"] == "from_config"
        assert runtime_env["OTHER"] == "kept"

    def test_addresses_the_local_dashboard_unless_ray_address_is_set(self, commands, monkeypatch):
        """RAY_ADDRESS already tells the ray CLI where to go; passing --address too would conflict."""
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")
        assert '--address="http://127.0.0.1:8265"' in _submit(commands)

        monkeypatch.setenv("RAY_ADDRESS", "http://10.0.0.1:8265")
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")
        assert "--address=" not in _submit(commands)

    def test_resolves_a_relative_train_script_against_the_repo(self, commands):
        """Launchers pass train.py, which only makes sense relative to the checkout."""
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert f"-- python3 {command_utils.repo_base_dir}/train.py " in _submit(commands)

    def test_keeps_an_absolute_train_script(self, commands):
        """An absolute path is already unambiguous and must not be rewritten."""
        _backend().execute_train(
            train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B", train_script="/opt/train.py"
        )

        assert "-- python3 /opt/train.py " in _submit(commands)


class TestBuildTrainEnvVars:
    @staticmethod
    def _request(**overrides):
        defaults = dict(
            train_args="",
            num_gpus_per_node=8,
            megatron_model_type="qwen3-4B",
            train_script="/opt/train.py",
            train_backend_fsdp=False,
            extra_env_vars={},
            megatron_path="/root/Megatron-LM",
            before_ray_job_submit=None,
            prepare_cmd={},
        )
        return ExecuteTrainRequest(**{**defaults, **overrides})

    def test_keeps_the_backend_vars_where_the_serialized_order_expects_them(self):
        """The ray backend serializes this dict verbatim, so the launch snapshots pin its key order."""
        env = common.train_env_vars(
            self._request(),
            {"NCCL_NVLS_ENABLE": "0", "MASTER_ADDR": "10.0.0.1"},
            config=command_utils.ExecuteTrainConfig(),
        )

        assert list(env) == ["PYTHONUNBUFFERED", "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_NVLS_ENABLE", "MASTER_ADDR"]

    def test_omits_the_connection_limit_for_fsdp(self):
        """Capping the connections breaks FSDP's computation and communication overlap."""
        env = common.train_env_vars(
            self._request(train_backend_fsdp=True), {}, config=command_utils.ExecuteTrainConfig()
        )

        assert "CUDA_DEVICE_MAX_CONNECTIONS" not in env

    def test_lets_the_caller_override_a_backend_var(self):
        """extra_env_vars is merged last so a script can win over anything the backend chose."""
        env = common.train_env_vars(
            self._request(extra_env_vars={"MASTER_ADDR": "caller"}),
            {"MASTER_ADDR": "backend"},
            config=command_utils.ExecuteTrainConfig(),
        )

        assert env["MASTER_ADDR"] == "caller"

    def test_lets_the_config_override_the_caller(self):
        """The operator's --extra-env-vars is the last word, above what the script hardcoded."""
        env = common.train_env_vars(
            self._request(extra_env_vars={"A": "from-script"}),
            {},
            config=command_utils.ExecuteTrainConfig(extra_env_vars="A=from-operator"),
        )

        assert env["A"] == "from-operator"

    def test_adds_the_coredump_vars_only_when_asked(self):
        """Core dumps are large, so they stay off until a run opts in."""
        config = command_utils.ExecuteTrainConfig(cuda_core_dump=True, output_dir="/runs")

        env = common.train_env_vars(self._request(), {}, config=config)

        assert env["CUDA_ENABLE_COREDUMP_ON_EXCEPTION"] == "1"
        assert env["CUDA_COREDUMP_FILE"] == "/runs/cuda_coredump_%h.%p.%t"
        assert "CUDA_ENABLE_COREDUMP_ON_EXCEPTION" not in common.train_env_vars(
            self._request(), {}, config=command_utils.ExecuteTrainConfig()
        )


class TestParseExtraEnvVars:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ('{"A": "1", "B": "2"}', {"A": "1", "B": "2"}),
            ("A=1 B=2", {"A": "1", "B": "2"}),
            ("", {}),
            ("   ", {}),
        ],
    )
    def test_accepts_json_and_shell_style(self, text, expected):
        """Operators pass either a JSON object or plain KEY=VALUE pairs."""
        assert common._parse_extra_env_vars(text) == expected


class TestCheckHasNvlink:
    @pytest.fixture
    def nvlink_probe(self, monkeypatch):
        def install(output: str) -> list[bool]:
            captured = []

            def fake_exec_command(cmd: str, capture_output: bool = False) -> str:
                captured.append(capture_output)
                return output

            patch_helper(monkeypatch, "exec_command_gpu", fake_exec_command)
            return captured

        return install

    def test_reports_true_when_links_are_counted(self, nvlink_probe):
        """A non-zero NVLink count from nvidia-smi means NVLink is present."""
        captured = nvlink_probe("4\n")

        assert _backend()._check_has_nvlink() is True
        assert captured == [True]

    def test_reports_false_without_links(self, nvlink_probe):
        """Without capture_output the real helper returns None and int(None) would abort the launch."""
        captured = nvlink_probe("0\n")

        assert _backend()._check_has_nvlink() is False
        assert captured == [True]


class TestGetDefaultWandbArgs:
    def test_is_empty_without_an_api_key(self, monkeypatch):
        """Unconfigured wandb must not inject half-populated flags."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)

        assert command_utils.get_default_wandb_args("tests/fast/utils/test_thing.py") == ""

    def test_names_the_project_after_the_test_file(self, monkeypatch):
        """The project name is how runs are found later, so it tracks the test file."""
        monkeypatch.setenv("WANDB_API_KEY", "secret")
        monkeypatch.delenv("GITHUB_COMMIT_NAME", raising=False)

        args = command_utils.get_default_wandb_args("tests/e2e/megatron/test_qwen3_4b.py", run_id="RUNID")

        assert "--use-wandb " in args
        assert "--wandb-project miles-test_qwen3_4b " in args
        assert "--wandb-group RUNID " in args
        assert "--wandb-key 'secret' " in args

    def test_qualifies_a_short_test_name_with_its_directory(self, monkeypatch):
        """Short stems like 'run.py' are ambiguous on their own."""
        monkeypatch.setenv("WANDB_API_KEY", "secret")

        args = command_utils.get_default_wandb_args("tests/e2e/megatron/run.py", run_id="RUNID")

        assert "--wandb-project miles-megatron_run " in args

    def test_decorates_the_group_with_commit_and_prefix(self, monkeypatch):
        """CI runs need the commit in the group name, and callers may add their own prefix."""
        monkeypatch.setenv("WANDB_API_KEY", "secret")
        monkeypatch.setenv("GITHUB_COMMIT_NAME", "abc123")

        args = command_utils.get_default_wandb_args("tests/e2e/megatron/test_qwen3_4b.py", "myprefix", run_id="RUNID")

        assert "--wandb-group myprefix_RUNID_abc123 " in args


class TestCreateRunId:
    def test_is_a_timestamp_with_a_random_suffix(self):
        """Concurrent runs on one machine must not collide on the run id."""
        date_part, time_part, random_part = command_utils.create_run_id().split("-")

        assert len(date_part) == 6 and date_part.isdigit()
        assert len(time_part) == 6 and time_part.isdigit()
        assert len(random_part) == 3 and random_part.isdigit()

    def test_varies_within_the_same_second(self):
        """Runs launched together in one second must not share a wandb group or dump directory."""
        suffixes = {command_utils.create_run_id().split("-")[2] for _ in range(50)}

        assert len(suffixes) > 1


class TestGetBoolEnvVar:
    @pytest.mark.parametrize(
        "value, expected",
        [("true", True), ("TRUE", True), ("1", True), ("false", False), ("0", False), ("maybe", False)],
    )
    def test_understands_the_usual_spellings(self, monkeypatch, value, expected):
        """Anything not recognizably truthy is treated as false rather than raising."""
        monkeypatch.setenv("SOME_FLAG", value)

        assert command_utils.get_bool_env_var("SOME_FLAG") is expected

    def test_falls_back_to_the_supplied_default(self, monkeypatch):
        """An unset variable takes the default, which is itself parsed as a string."""
        monkeypatch.delenv("SOME_FLAG", raising=False)

        assert command_utils.get_bool_env_var("SOME_FLAG") is False
        assert command_utils.get_bool_env_var("SOME_FLAG", "1") is True


class TestGetEnvEnableInfiniteRun:
    def test_defaults_to_off(self, monkeypatch):
        """Infinite runs must be opt-in; a stuck CI job is expensive."""
        monkeypatch.delenv("MILES_TEST_ENABLE_INFINITE_RUN", raising=False)
        assert command_utils.get_env_enable_infinite_run() is False

        monkeypatch.setenv("MILES_TEST_ENABLE_INFINITE_RUN", "true")
        assert command_utils.get_env_enable_infinite_run() is True


class TestEncodePseudoFile:
    def test_round_trips_through_resolve_file_arg(self):
        """The encoded argument is what the training process will be asked to resolve."""
        encoded = command_utils.encode_pseudo_file("hello: world")

        assert resolve_file_arg(encoded) == "hello: world"

    def test_is_deterministic(self):
        """A hot restart must recompute the identical launch command."""
        assert command_utils.encode_pseudo_file("hello: world") == command_utils.encode_pseudo_file("hello: world")

    def test_survives_a_command_line_round_trip(self):
        """The value is interpolated into a shell command, so it must not need quoting."""
        encoded = command_utils.encode_pseudo_file("a: 1\nb: 'two words'\n")

        assert shlex.split(f"--custom-config-path {encoded}")[1] == encoded


def _hardware_launchers_accept() -> set[str]:
    """Every value any launcher's `--hardware` takes, minus the sentinel that stands for "ask the node"."""
    return set().union(*launcher_hardware_literals().values()) - {"auto"}


class TestHardwareTables:
    def test_every_hardware_a_launcher_accepts_declares_its_gpus_per_node(self):
        """A launcher whose hardware is missing here raises KeyError before doing anything."""
        assert _hardware_launchers_accept() <= command_utils.NUM_GPUS_OF_HARDWARE.keys()

    def test_every_hardware_with_a_generation_also_has_a_gpu_count(self):
        """Every launcher reads the GPU count, while only some read the generation."""
        assert command_utils.GENERATION_HARDWARE.keys() <= command_utils.NUM_GPUS_OF_HARDWARE.keys()


def _fake_torch(monkeypatch, *, capability, machine, total_memory=141 * 1024**3, name="fake", hip=None):
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            version=SimpleNamespace(hip=hip),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda: name,
                get_device_capability=lambda: capability,
                get_device_properties=lambda device: SimpleNamespace(total_memory=total_memory),
            ),
        ),
    )


class TestDetectHardware:
    @pytest.mark.parametrize(
        ("capability", "machine", "expected"),
        [
            ((10, 0), "x86_64", "B200"),
            ((10, 0), "aarch64", "GB200"),
            ((10, 3), "x86_64", "B300"),
            ((10, 3), "aarch64", "GB300"),
        ],
    )
    def test_grace_is_what_separates_the_superchip_from_the_pcie_part(
        self, monkeypatch, capability, machine, expected
    ):
        """GB200/GB300 carry the same die as B200/B300, so the host CPU is the only signal."""
        _fake_torch(monkeypatch, capability=capability, machine=machine)

        assert command_utils.detect_hardware() == expected

    @pytest.mark.parametrize(("total_memory", "expected"), [(80 * 1024**3, "H100"), (141 * 1024**3, "H200")])
    def test_hopper_is_split_by_memory(self, monkeypatch, total_memory, expected):
        """H100 and H200 share sm90; only the HBM capacity tells them apart."""
        _fake_torch(monkeypatch, capability=(9, 0), machine="x86_64", total_memory=total_memory)

        assert command_utils.detect_hardware() == expected

    def test_rocm_reads_the_device_name(self, monkeypatch):
        _fake_torch(monkeypatch, capability=(9, 5), machine="x86_64", name="AMD Instinct MI355X", hip="6.4")

        assert command_utils.detect_hardware() == "MI355X"

    def test_an_unrecognized_device_asks_for_an_explicit_hardware(self, monkeypatch):
        """Silently guessing the wrong profile costs a whole run; a launcher flag is one word."""
        _fake_torch(monkeypatch, capability=(12, 0), machine="x86_64", name="NVIDIA GeForce RTX 5090")

        with pytest.raises(AssertionError, match="pass --hardware"):
            command_utils.detect_hardware()

    def test_every_detectable_hardware_is_a_table_entry(self, monkeypatch):
        """A detected name the tables do not carry is a KeyError at the first lookup."""
        for capability in ((9, 0), (10, 0), (10, 3)):
            for machine in ("x86_64", "aarch64"):
                _fake_torch(monkeypatch, capability=capability, machine=machine)
                assert command_utils.detect_hardware() in command_utils.NUM_GPUS_OF_HARDWARE
