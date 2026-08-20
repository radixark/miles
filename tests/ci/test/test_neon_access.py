import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JOB = load_module("neon_access_job", ROOT / ".github/workflows/scripts/neon_access_job.py")
CLIENT = load_module(
    "run_neon_workflow",
    ROOT / ".claude/skills/neon-access/scripts/run_neon_workflow.py",
)


class FakeResult:
    def __init__(self, statusmessage, columns=(), rows=()):
        self.statusmessage = statusmessage
        self.description = [SimpleNamespace(name=column) for column in columns] if columns else None
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


class FakeCursor:
    def __init__(self, result_sets, error=None):
        self.result_sets = result_sets
        self.error = error
        self.executed = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql_text, *, prepare):
        self.executed = (sql_text, prepare)
        if self.error is not None:
            raise self.error

    def results(self):
        return iter(self.result_sets)


class FakeConnection:
    def __init__(self, cursor):
        self.fake_cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return self.fake_cursor


class FakeDriver:
    class Error(Exception):
        pass

    def __init__(self, cursor):
        self.fake_connection = FakeConnection(cursor)
        self.connect_call = None

    def connect(self, dsn, **kwargs):
        self.connect_call = (dsn, kwargs)
        return self.fake_connection


def execute(driver, sql_text, *, max_result_bytes=4096):
    return JOB.execute_request(
        driver,
        dsn="postgresql://not-used",
        sql_text=sql_text,
        request_id="00000000-0000-0000-0000-000000000001",
        run_id="123",
        actor="writer",
        reason="offline test",
        max_result_bytes=max_result_bytes,
    )


def test_arbitrary_multi_statement_sql_is_passed_once_and_unchanged():
    sql_text = """CREATE TEMP TABLE t(v text);
INSERT INTO t VALUES ('do not split; this');
DO $$ BEGIN RAISE NOTICE 'inside;body'; END $$;
SELECT v FROM t;
"""
    cursor = FakeCursor(
        [
            FakeResult("CREATE TABLE"),
            FakeResult("INSERT 0 1"),
            FakeResult("DO"),
            FakeResult("SELECT 1", ["v"], [("do not split; this",)]),
        ]
    )
    driver = FakeDriver(cursor)

    result = execute(driver, sql_text)

    assert cursor.executed == (sql_text, False)
    assert driver.connect_call == (
        "postgresql://not-used",
        {"autocommit": True, "prepare_threshold": None},
    )
    assert [item["command_status"] for item in result["results"]] == [
        "CREATE TABLE",
        "INSERT 0 1",
        "DO",
        "SELECT 1",
    ]
    assert result["results"][-1]["rows"] == [["do not split; this"]]
    assert result["sql_sha256"] == hashlib.sha256(sql_text.encode()).hexdigest()


def test_result_encoding_and_limit_do_not_change_execution():
    cursor = FakeCursor(
        [
            FakeResult(
                "SELECT 3",
                ["payload"],
                [
                    (b"binary",),
                    ({"nested": (1, 2)},),
                    ("x" * 100,),
                ],
            )
        ]
    )
    driver = FakeDriver(cursor)

    result = execute(driver, "SELECT arbitrary_result", max_result_bytes=70)

    assert cursor.executed == ("SELECT arbitrary_result", False)
    assert result["results"][0]["rows"] == [
        [{"type": "bytes", "base64": "YmluYXJ5"}],
        [{"nested": [1, 2]}],
    ]
    assert result["truncated"] is True
    assert result["status"] == "ok"


def test_database_error_is_returned_in_result():
    error = FakeDriver.Error("permission denied by PostgreSQL")
    error.sqlstate = "42501"
    driver = FakeDriver(FakeCursor([], error=error))

    result = execute(driver, "DROP TABLE protected")

    assert result["status"] == "error"
    assert result["error"] == {
        "type": "Error",
        "sqlstate": "42501",
        "message": "permission denied by PostgreSQL",
    }


def test_sql_transport_round_trip_preserves_utf8_bytes():
    sql_bytes = "SELECT '雪';\n-- exact trailing newline\n".encode()

    packed, digest = CLIENT.pack_sql(sql_bytes)

    assert JOB.decode_sql(packed).encode() == sql_bytes
    assert digest == hashlib.sha256(sql_bytes).hexdigest()


def test_cms_round_trip_and_identity_validation(tmp_path):
    key_path, cert_path, _ = CLIENT._create_recipient(tmp_path)
    result = {
        "schema_version": 1,
        "request_id": "request",
        "run_id": "123",
        "actor": "writer",
        "sql_sha256": "digest",
        "status": "ok",
    }
    source_path = tmp_path / "result.json"
    encrypted_path = tmp_path / "result.json.cms"
    decrypted_path = tmp_path / "decrypted.json"
    source_path.write_text(json.dumps(result), encoding="utf-8")
    subprocess.run(
        [
            "openssl",
            "cms",
            "-encrypt",
            "-binary",
            "-aes256",
            "-in",
            str(source_path),
            "-outform",
            "DER",
            "-out",
            str(encrypted_path),
            str(cert_path),
        ],
        check=True,
    )

    CLIENT.decrypt_result(
        encrypted_path,
        decrypted_path,
        key_path,
        cert_path,
    )
    decrypted = json.loads(decrypted_path.read_text(encoding="utf-8"))
    CLIENT.validate_result(
        decrypted,
        request_id="request",
        run_id="123",
        actor="writer",
        sql_sha256="digest",
    )
    assert decrypted == result


def test_workflow_keeps_secret_and_sql_at_executor_boundary():
    workflow = (ROOT / ".github/workflows/neon-access.yml").read_text()

    assert workflow.count("secrets.NEON_DATABASE_URL") == 1
    assert "SQL_GZIP_BASE64: ${{ inputs.sql_gzip_base64 }}" in workflow
    assert "run: python3 .github/workflows/scripts/neon_access_job.py" in workflow
    assert "GITHUB_STEP_SUMMARY" not in workflow
    assert "collaborators/${actor}/permission" in workflow
    assert "github.triggering_actor" in workflow
    assert 'permission" != "write"' in workflow
    assert "Neon access must run from the default branch" in workflow
    assert "retention-days: 1" in workflow
    assert "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683" in workflow
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in workflow
