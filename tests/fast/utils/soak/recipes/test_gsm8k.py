import os
from pathlib import Path

import pytest
from tests.fast.utils.soak.recipes.recipe_fakes import _FakeBackend
from tests.utils.ft.launch import MEGATRON_PATH
from tests.utils.soak.core.utils import DATA_DIR, MODEL_DIR, compute_base_url
from tests.utils.soak.recipes import gsm8k
from tests.utils.soak.recipes.gsm8k import (
    MODEL_NAME,
    MODEL_TYPE,
    ROLLOUT_GPUS,
    TRAIN_GPUS,
    Gsm8kRun,
    get_gsm8k_train_args,
    prepare_gsm8k,
    prepare_gsm8k_run,
)

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend


def _config() -> ExecuteTrainConfig:
    return ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, namespace="rl", run_id="run-a")


@pytest.fixture
def backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeBackend:
    backend = _FakeBackend()
    monkeypatch.setenv("MILES_TEST_DUMPS_ROOT", str(tmp_path / "dumps"))
    monkeypatch.setattr(gsm8k, "create_backend_for_run", lambda config: backend)
    return backend


def _prepare(**overrides: object) -> Gsm8kRun:
    return prepare_gsm8k_run(
        **{
            "config": _config(),
            "test_name": "soak_a",
            "seed": 7,
            "num_rollout": 30,
            "build_extra_train_args": lambda dump_dir: f"--save {dump_dir}/checkpoints ",
            **overrides,
        }
    )


def _value_of(train_args: str, flag: str) -> str:
    tokens = train_args.split()
    return tokens[tokens.index(flag) + 1]


class TestPrepareGsm8k:
    def test_the_model_is_downloaded_converted_and_the_dataset_fetched(self) -> None:
        """The run reads the converted torch_dist checkpoint and gsm8k parquet the backend prepared."""
        backend = _FakeBackend()

        prepare_gsm8k(backend)

        assert backend.calls == [
            ("exec_command_cpu", f"mkdir -p {MODEL_DIR} {DATA_DIR}"),
            ("exec_command_cpu", f"hf download Qwen/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}"),
            (
                "convert_checkpoint",
                {
                    "model_name": MODEL_NAME,
                    "megatron_model_type": MODEL_TYPE,
                    "num_gpus_per_node": TRAIN_GPUS,
                    "hf_checkpoint": f"{MODEL_DIR}/{MODEL_NAME}",
                    "dir_dst": MODEL_DIR,
                    "megatron_path": MEGATRON_PATH,
                },
            ),
            ("hf_download_dataset", ("zhuzilin/gsm8k", DATA_DIR)),
        ]


class TestGetGsm8kTrainArgs:
    def test_the_run_length_threshold_and_paths_reach_the_arguments(self) -> None:
        """The launched run trains the requested rollouts on the prepared model and grades at the threshold."""
        train_args = get_gsm8k_train_args(seed=7, num_rollout=30, test_name="soak_a", metric_threshold=0.61)

        assert _value_of(train_args, "--num-rollout") == "30"
        assert _value_of(train_args, "--ci-metric-checker-threshold") == "0.61"
        assert _value_of(train_args, "--hf-checkpoint") == f"{MODEL_DIR}/{MODEL_NAME}/"
        assert _value_of(train_args, "--ref-load") == f"{MODEL_DIR}/{MODEL_NAME}_torch_dist"
        assert _value_of(train_args, "--prompt-data") == f"{DATA_DIR}/gsm8k/train.parquet"
        assert _value_of(train_args, "--rollout-num-gpus") == str(ROLLOUT_GPUS)
        assert _value_of(train_args, "--actor-num-gpus-per-node") == str(TRAIN_GPUS)

    def test_fault_tolerance_flags_follow_the_switch_while_the_api_server_stays(self) -> None:
        """A hot-restart run keeps the cell API and p2p updates but must not enable training fault tolerance."""
        enabled = get_gsm8k_train_args(seed=7, num_rollout=30, test_name="t").split()
        disabled = get_gsm8k_train_args(seed=7, num_rollout=30, test_name="t", enable_fault_tolerance=False).split()

        assert {"--use-fault-tolerance", "--mini-ft-controller-enable"} <= set(enabled)
        assert _value_of(" ".join(enabled), "--ft-components") == "train"
        assert not {"--use-fault-tolerance", "--mini-ft-controller-enable", "--ft-components"} & set(disabled)
        for args in (enabled, disabled):
            assert "--api-server-port" in args
            assert _value_of(" ".join(args), "--update-weight-transfer-mode") == "p2p"

    def test_only_a_fully_async_run_gets_the_fully_async_flags(self) -> None:
        """The sync driver must not receive fully-async arguments it would silently ignore."""
        assert "--fully-async" in get_gsm8k_train_args(seed=7, num_rollout=30, test_name="t", fully_async=True).split()
        assert "--fully-async" not in get_gsm8k_train_args(seed=7, num_rollout=30, test_name="t").split()


class TestPrepareGsm8kRun:
    def test_the_run_is_bound_to_one_fresh_dump_dir_and_its_evidence(
        self, tmp_path: Path, backend: _FakeBackend
    ) -> None:
        """Events, extra args, evidence and the launch spec all hang off one dump dir under this run id."""
        run = _prepare(fully_async=True)

        dump_dir = tmp_path / "dumps" / "run-a" / "soak_a"
        assert run.dump_dir == str(dump_dir)
        assert run.events_dir == dump_dir / EVENTS_DIRNAME
        assert _value_of(run.launch_spec.train_args, "--save-debug-event-data") == f"{dump_dir}/{EVENTS_DIRNAME}"
        assert _value_of(run.launch_spec.train_args, "--save") == f"{dump_dir}/checkpoints"
        assert run.launch_spec.fully_async
        assert run.launch_spec.config == _config()
        assert run.evidence_dir.parent == dump_dir.with_name("soak_a-soak")
        assert run.event_log.path == run.evidence_dir / "events.jsonl" and run.event_log.path.is_file()
        assert run.base_url == compute_base_url(run.launch_spec.config)
        assert backend.calls, "the model and dataset were never prepared"

    def test_a_dump_dir_holding_a_previous_run_is_refused(self, tmp_path: Path, backend: _FakeBackend) -> None:
        """Reusing a run id would grade this run on another run's leftover events."""
        stale = tmp_path / "dumps" / "run-a" / "soak_a" / EVENTS_DIRNAME
        stale.mkdir(parents=True)

        with pytest.raises(ValueError, match="existing artifacts"):
            _prepare()

    def test_the_launching_shell_proxies_are_dropped(
        self, backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A proxied local API request would never reach the run the soak drives."""
        for name in ("http_proxy", "HTTPS_PROXY"):
            monkeypatch.setenv(name, "http://proxy:1")

        _prepare()

        assert "http_proxy" not in os.environ and "HTTPS_PROXY" not in os.environ
