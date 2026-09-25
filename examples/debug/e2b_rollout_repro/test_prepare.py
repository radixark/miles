"""Check rendering with paths containing spaces; no GPU or service access."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    names = json.loads((here / "train-task-names.json").read_text())
    with tempfile.TemporaryDirectory(prefix="e2b repro ") as directory:
        root = Path(directory).resolve()
        for name in names:
            task = root / "tasks" / name
            task.mkdir(parents=True)
            (task / "instruction.md").write_text("Synthetic packaging fixture.")
        manifest = root / "templates.json"
        manifest.write_text(json.dumps({"results": [{"task": name} for name in names]}))
        key = root / "key"
        key.write_text("TEST_ONLY_NOT_A_CREDENTIAL")
        output = root / "output"
        command = [sys.executable, str(here / "prepare.py")]
        args = {"output": output, "model": root, "reference": root, "tasks": root / "tasks",
                "harbor": root, "megatron": root, "dependencies": root, "templates": manifest,
                "key_file": key, "api_url": "https://example.invalid", "sandbox_url": "http://example.invalid"}
        for flag, value in args.items():
            command.extend(["--" + flag, str(value)])
        subprocess.run(command, check=True)
        request = json.loads((output / "ray-job-request.json").read_text())
        env = request["runtime_env"]["env_vars"]
        assert env["PILOT_ROOT"] == str(output)
        assert env["E2B_API_KEY_FILE"] == str(key)
        assert "TEST_ONLY_NOT_A_CREDENTIAL" not in json.dumps(request)
        assert env["SANDBOX_CONCURRENCY"] == "128"
        assert "${" not in json.dumps(request)
        assert len((output / "train.jsonl").read_text().splitlines()) == 60
        assert "--debug-rollout-only" in request["entrypoint"]
    print("PREPARE_TEST_PASSED")


if __name__ == "__main__":
    main()
