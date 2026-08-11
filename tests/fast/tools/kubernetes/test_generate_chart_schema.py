import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
GENERATOR_PATH = REPO_ROOT / "tools" / "kubernetes" / "generate_chart_schema.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_chart_schema", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return load_generator()


@pytest.fixture(scope="module")
def schemas(generator) -> dict[Path, str]:
    return generator.generated_schemas()


def _validator(schema: dict[str, Any]):
    jsonschema = pytest.importorskip("jsonschema")
    return jsonschema.Draft7Validator(schema)


def _run_schema(schemas: dict[Path, str]) -> dict[str, Any]:
    return json.loads(schemas[Path("charts/miles-run/values.schema.json")])


def _minimal_run_values() -> dict[str, Any]:
    return {
        "infra": {
            "image": {"repository": "registry.local/miles", "tag": "v1"},
            "sharedStorage": {"type": "none", "mountPath": "/cluster-storage"},
        },
        "run": {
            "id": "260101-000000-000",
            "stateFile": "/cluster-storage/miles_data/miles-runs/260101-000000-000/state/orchestrator.state",
            "objectNames": {
                "orchestrator": "r-miles-run-orchestrator",
                "mooncakeMaster": "r-miles-run-mooncake-master",
                "colocatePairing": "r-miles-run-colocate-pairing",
                "uninstall": "r-miles-run-uninstall",
                "uninstallManifest": "r-miles-run-uninstall-manifest",
            },
        },
    }


class TestGeneratedFilesAreCommitted:
    def test_every_schema_on_disk_is_what_the_generator_produces(self, schemas):
        """A hand-edited schema drifts from the types the launcher builds values with, silently."""
        for path, content in schemas.items():
            assert (REPO_ROOT / path).read_text() == content, path

    def test_the_generator_writes_exactly_the_three_schemas_the_repo_ships(self, schemas):
        """A chart whose schema nobody generates would keep a stale hand-written one forever."""
        assert set(schemas) == {
            Path("charts/shared-infra.schema.json"),
            Path("charts/miles-run/values.schema.json"),
            Path("charts/miles-workbench/values.schema.json"),
        }

    def test_every_schema_declares_the_dialect_helm_validates_with(self, schemas):
        """helm uses gojsonschema, which reads draft-07; pydantic emits 2020-12 unless it is converted."""
        for path, content in schemas.items():
            assert json.loads(content)["$schema"] == "http://json-schema.org/draft-07/schema#", path


class TestRunSchemaAcceptsRealValues:
    def test_the_values_the_launcher_generates_validate(self, schemas):
        """The schema exists to catch bad values, and rejecting the good ones would block every run."""
        _validator(_run_schema(schemas)).validate(_minimal_run_values())

    def test_the_charts_own_defaults_validate(self, schemas):
        """helm merges values.yaml under every install, so defaults that fail the schema break all of them."""
        yaml = pytest.importorskip("yaml")
        defaults = yaml.safe_load((REPO_ROOT / "charts" / "miles-run" / "values.yaml").read_text())

        _validator(_run_schema(schemas)).validate(defaults)

    def test_a_pool_carrying_every_optional_key_validates(self, schemas):
        """Optional keys the builder emits only sometimes must not be rejected when it does emit them."""
        values = _minimal_run_values()
        values["run"]["trainerEngines"] = [
            {
                "name": "trainer-engine-actor",
                "objectName": "r-miles-run-trainer-engine-actor",
                "poolId": "trainer-engine-actor",
                "command": ["python", "-m", "serve"],
                "ports": [{"name": "master", "port": 9000}],
                "env": {"NCCL_CUMEM_ENABLE": "0"},
                "meta": {"miles.radixark.io/gpu-ids": "0,1"},
                "replicas": 2,
                "size": 2,
                "resources": {"limits": {"nvidia.com/gpu": 8}},
            }
        ]

        _validator(_run_schema(schemas)).validate(values)


class TestRunSchemaRejectsBadValues:
    def test_a_typo_in_a_section_the_chart_owns_is_rejected(self, schemas):
        """A silently ignored typo is exactly the failure a strict schema is installed to prevent."""
        jsonschema = pytest.importorskip("jsonschema")
        values = _minimal_run_values()
        values["run"]["objectNamez"] = {}

        with pytest.raises(jsonschema.ValidationError):
            _validator(_run_schema(schemas)).validate(values)

    def test_an_enabled_command_job_without_a_command_is_rejected(self, schemas):
        """The conditional is the only thing standing between an enabled job and a pod that runs nothing."""
        jsonschema = pytest.importorskip("jsonschema")
        values = _minimal_run_values()
        values["commandJob"] = {"enabled": True, "name": "convert", "objectName": "c"}

        with pytest.raises(jsonschema.ValidationError):
            _validator(_run_schema(schemas)).validate(values)

    def test_a_pythonpath_override_in_the_shared_env_is_rejected(self, schemas):
        """miles composes PYTHONPATH from the mounted checkouts; a user value would hide half of them."""
        jsonschema = pytest.importorskip("jsonschema")
        values = _minimal_run_values()
        values["infra"]["env"] = {"PYTHONPATH": "/somewhere"}

        with pytest.raises(jsonschema.ValidationError):
            _validator(_run_schema(schemas)).validate(values)

    def test_a_shared_host_path_mount_without_the_path_is_rejected(self, schemas):
        """A hostPath mount with no host path renders and installs, then leaves every pod unable to start."""
        jsonschema = pytest.importorskip("jsonschema")
        values = _minimal_run_values()
        values["infra"]["sharedStorage"] = {"type": "hostPath", "mountPath": "/cluster-storage"}

        with pytest.raises(jsonschema.ValidationError):
            _validator(_run_schema(schemas)).validate(values)
