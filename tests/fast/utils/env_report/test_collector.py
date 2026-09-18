import json
import os
import subprocess
import sys
import types
from unittest.mock import patch


from tests.fast.utils.env_report.conftest import SAMPLE_PIP_INSPECT, make_args

from miles.utils.audit_utils.event_logger.models import EnvReport, EnvReportEditablePackageInfo
from miles.utils.env_report.collector import (
    _collect_key_versions,
    _collect_pip_info,
    _dump_args,
    _is_editable,
    _parse_pip_entry,
    collect_env_report,
    collect_env_report_snapshot,
)
from miles.utils.env_report.launcher_report import LAUNCHER_REPORT_ENV_VAR


class TestParsePipEntry:
    def test_normal_package(self) -> None:
        entry = _parse_pip_entry({"metadata": {"name": "torch", "version": "2.5.0"}})
        assert entry == {"name": "torch", "version": "2.5.0"}

    def test_missing_metadata(self) -> None:
        entry = _parse_pip_entry({})
        assert entry == {"name": "", "version": ""}


class TestIsEditable:
    def test_editable_package(self) -> None:
        pkg = {"direct_url": {"url": "file:///workspace/miles", "dir_info": {"editable": True}}}
        assert _is_editable(pkg) is True

    def test_non_editable_package(self) -> None:
        assert _is_editable({"metadata": {"name": "torch"}}) is False

    def test_archive_url_not_editable(self) -> None:
        pkg = {"direct_url": {"url": "https://example.com/foo.tar.gz", "archive_info": {}}}
        assert _is_editable(pkg) is False


class TestCollectPipInfo:
    def test_parses_pip_inspect_output(self) -> None:
        mock_result = subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=0,
            stdout=json.dumps(SAMPLE_PIP_INSPECT),
            stderr="",
        )
        with patch("miles.utils.env_report.collector.subprocess.run", return_value=mock_result):
            editable, full_list = _collect_pip_info({})

        assert len(full_list) == 4
        assert full_list[0] == {"name": "miles", "version": "0.2.1"}
        assert full_list[2] == {"name": "torch", "version": "2.5.0"}

        assert len(editable) == 2
        assert editable[0] == EnvReportEditablePackageInfo(
            name="miles",
            version="0.2.1",
            location="/workspace/miles",
        )
        assert editable[1] == EnvReportEditablePackageInfo(
            name="sglang",
            version="0.4.0",
            location="/workspace/sglang",
        )

    def test_pip_inspect_failure_returns_empty(self) -> None:
        mock_result = subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=1,
            stdout="",
            stderr="error",
        )
        with patch("miles.utils.env_report.collector.subprocess.run", return_value=mock_result):
            editable, full_list = _collect_pip_info({})
        assert editable == []
        assert full_list == []

    def test_pip_inspect_exception_returns_empty(self) -> None:
        with patch("miles.utils.env_report.collector.subprocess.run", side_effect=OSError("no pip")):
            editable, full_list = _collect_pip_info({})
        assert editable == []
        assert full_list == []

    def test_runs_pip_with_the_environment_it_is_given(self) -> None:
        """The caller hands in a snapshot; reading os.environ here would race whoever mutates it."""
        mock_result = subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=0,
            stdout=json.dumps(SAMPLE_PIP_INSPECT),
            stderr="",
        )
        with patch("miles.utils.env_report.collector.subprocess.run", return_value=mock_result) as mock_run:
            _collect_pip_info({"HOME": "/root"})

        assert mock_run.call_args.kwargs["env"] == {"HOME": "/root"}


class TestCollectProcessEnvSnapshot:
    def test_excludes_pythonpath_from_the_pip_probe(self) -> None:
        """pip inspect misses editable packages whose source is on the PYTHONPATH."""
        with patch.dict(os.environ, {"PYTHONPATH": "/workspace/Megatron-LM"}):
            snapshot = collect_env_report_snapshot(make_args())

        assert "PYTHONPATH" not in snapshot.probe_env

    def test_snapshots_the_environment_the_probe_will_run_with(self) -> None:
        """The probe runs on another thread, while the caller is still setting RANK and friends."""
        with patch.dict(os.environ, {"MILES_TEST_ENV_REPORT_MARK": "before"}):
            snapshot = collect_env_report_snapshot(make_args())
            os.environ["MILES_TEST_ENV_REPORT_MARK"] = "after"

            assert snapshot.probe_env["MILES_TEST_ENV_REPORT_MARK"] == "before"


class TestDumpArgs:
    def test_keeps_serializable_values(self) -> None:
        dump = _dump_args(make_args(lr=1e-4, tags=["a"], nested={"x": 1}, flag=True, missing=None))
        assert dump.skipped_names == []
        assert dump.values["lr"] == 1e-4
        assert dump.values["tags"] == ["a"]
        assert dump.values["nested"] == {"x": 1}
        assert dump.values["missing"] is None

    def test_skips_unserializable_values(self) -> None:
        """A non-JSON arg is skipped by name instead of being coerced to a lossy string."""
        dump = _dump_args(make_args(model=object(), lr=1.0))
        assert dump.skipped_names == ["model"]
        assert "model" not in dump.values
        assert dump.values["lr"] == 1.0

    def test_redacts_a_declared_secret_arg(self) -> None:
        dump = _dump_args(make_args(wandb_key="abc"))
        assert dump.values["wandb_key"].startswith("redacted-sha256:")

    def test_keeps_an_unset_secret_arg_as_none(self) -> None:
        """--wandb-key defaults to None, and hashing that would crash every process at startup."""
        assert _dump_args(make_args(wandb_key=None)).values["wandb_key"] is None

    def test_redacts_every_entry_of_a_list_valued_secret_arg(self) -> None:
        """The router's control-plane keys arrive as a list, which a string-only redaction would pass through."""
        dump = _dump_args(make_args(router_control_plane_api_keys=["k1:n:r:s1", "k2:n:r:s2"]))

        assert all(entry.startswith("redacted-sha256:") for entry in dump.values["router_control_plane_api_keys"])

    def test_redacts_the_secrets_inside_an_environment_valued_arg(self) -> None:
        """--train-env-vars is a whole environment, so it is redacted by variable name, not by arg name."""
        dump = _dump_args(make_args(train_env_vars={"WANDB_API_KEY": "hunter2", "NCCL_DEBUG": "INFO"}))

        assert dump.values["train_env_vars"]["WANDB_API_KEY"].startswith("redacted-sha256:")
        assert dump.values["train_env_vars"]["NCCL_DEBUG"] == "INFO"

    def test_snapshots_nested_values_instead_of_referencing_them(self) -> None:
        """The dump outlives the caller, which is free to keep mutating its own args."""
        tags = ["a"]
        dump = _dump_args(make_args(tags=tags))
        tags.append("b")
        assert dump.values["tags"] == ["a"]

    def test_keeps_dataset_column_args_that_merely_end_in_key(self) -> None:
        """--reward-key names a dataset column; hashing it would hide the run's actual configuration."""
        dump = _dump_args(make_args(reward_key="reward", input_key="prompt"))
        assert dump.values["reward_key"] == "reward"
        assert dump.values["input_key"] == "prompt"

    def test_dump_is_json_serializable(self) -> None:
        dump = _dump_args(make_args(model=object(), lr=1.0))
        assert json.loads(json.dumps(dump.model_dump()))["values"]["lr"] == 1.0


class TestCollectKeyVersions:
    def test_reports_python_and_known_packages(self) -> None:
        versions = _collect_key_versions(
            [{"name": "torch", "version": "2.5.0"}, {"name": "SGLang", "version": "0.4.0"}]
        )
        assert versions["python"] == ".".join(str(part) for part in sys.version_info[:3])
        assert versions["sglang"] == "0.4.0"
        assert "platform" in versions

    def test_ignores_unknown_packages(self) -> None:
        versions = _collect_key_versions([{"name": "numpy", "version": "1.26.0"}])
        assert "numpy" not in versions

    def test_reports_torch_cuda_when_torch_is_imported(self) -> None:
        torch = types.SimpleNamespace(__version__="2.5.0", version=types.SimpleNamespace(cuda="12.4"))
        with patch.dict(sys.modules, {"torch": torch}):
            versions = _collect_key_versions([])

        assert versions["torch"] == "2.5.0"
        assert versions["torch_cuda"] == "12.4"

    def test_reports_an_empty_cuda_version_for_a_cpu_torch(self) -> None:
        torch = types.SimpleNamespace(__version__="2.5.0", version=types.SimpleNamespace(cuda=None))
        with patch.dict(sys.modules, {"torch": torch}):
            assert _collect_key_versions([])["torch_cuda"] == ""

    def test_reports_nothing_about_an_unimported_torch(self) -> None:
        """torch is read from sys.modules, so an unimported torch costs nothing and reports nothing."""
        with patch.dict(sys.modules, {"torch": None}):
            assert "torch_cuda" not in _collect_key_versions([])


class TestCollectEnvReport:
    def _collect(self, **overrides) -> EnvReport:
        return collect_env_report(snapshot=collect_env_report_snapshot(make_args(**overrides)))

    def _with_record(self, tmp_path, record: str = '{"flavor": "test"}', **overrides) -> EnvReport:
        path = tmp_path / "launch-1.json"
        path.write_text(record)
        return self._collect(env_report=str(path), **overrides)

    def test_returns_structured_report(self, mocked_pip_inspect, tmp_path) -> None:
        report = self._with_record(tmp_path)

        assert isinstance(report, EnvReport)
        assert report.process.launcher_env_report == {"flavor": "test"}
        assert len(report.editable_packages) == 2
        assert len(report.full_pip_list) == 4

    def test_records_process_identity_context(self, mocked_pip_inspect) -> None:
        """The audit needs to know which host and command line produced this report."""
        report = self._collect(lr=1.0)
        assert report.process.hostname
        assert report.process.argv == sys.argv
        assert report.process.args.values["lr"] == 1.0

    def test_records_redacted_environment(self, mocked_pip_inspect) -> None:
        with patch.dict(os.environ, {"MILES_TEST_ENV_REPORT_TOKEN": "hunter2", "MILES_TEST_ENV_REPORT_RANK": "7"}):
            report = self._collect()

        assert report.process.env_vars["MILES_TEST_ENV_REPORT_RANK"] == "7"
        assert "hunter2" not in report.process.env_vars["MILES_TEST_ENV_REPORT_TOKEN"]

    def test_leaves_the_launcher_record_out_of_the_environment_dump(self, mocked_pip_inspect, tmp_path) -> None:
        """The record is already stored in full, so naming its path twice only adds noise."""
        path = tmp_path / "launch-1.json"
        with patch.dict(os.environ, {LAUNCHER_REPORT_ENV_VAR: str(path)}):
            report = self._with_record(tmp_path)

        assert LAUNCHER_REPORT_ENV_VAR not in report.process.env_vars
        assert report.process.launcher_env_report == {"flavor": "test"}

    def test_records_key_versions(self, mocked_pip_inspect) -> None:
        report = self._collect()
        assert report.key_versions["sglang"] == "0.4.0"

    def test_a_run_with_no_launcher_records_no_launcher_report(self, mocked_pip_inspect) -> None:
        """A run started by hand names no record, and that is not a failure."""
        assert self._collect(env_report="").process.launcher_env_report is None

    def test_a_record_that_cannot_be_read_leaves_the_rest_of_the_report_intact(
        self, mocked_pip_inspect, tmp_path
    ) -> None:
        """Everything else in the report is still worth having when the shared disk lost the record."""
        report = self._collect(env_report=str(tmp_path / "gone.json"))

        assert report.process.launcher_env_report is None
        assert report.process.hostname

    def test_report_serializable(self, mocked_pip_inspect, tmp_path) -> None:
        report = self._with_record(tmp_path, record='{"x": 1}', model=object())
        parsed = json.loads(report.model_dump_json())
        assert parsed["editable_packages"][0]["name"] == "miles"
        assert parsed["process"]["args"]["skipped_names"] == ["model"]
