"""Run a connectivity/integrity gate followed by the controlled payload stress test."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from tap import Tap


class Args(Tap):
    root: str
    parent: str
    fixture: str


def verify(root: Path) -> None:
    results = json.loads((root / "results.json").read_text())
    rows = [row for result in results for row in result["rows"]]
    if root.name == "gate":
        assert all("error" not in row and row.get("sandbox_ok") for row in rows), rows
    assert len({row["training_hash"] for row in rows if "training_hash" in row}) == 1
    candidate_hashes = {row["candidate_hash"] for row in rows if "candidate_hash" in row}
    assert len(candidate_hashes) == 1
    assert all(value == "True" or value == "None" for value in json.loads((root / "cleanup.json").read_text()))


def main() -> None:
    args = Args().parse_args()
    root = Path(args.root)
    env = os.environ.copy()
    env.update(json.loads((Path(args.parent) / "full-ray-job-request.json").read_text())["runtime_env"]["env_vars"])
    script = Path(__file__).with_name("payload_control.py")
    for phase, extra in [("gate", ["--workers", "1", "--concurrency", "1", "--sandbox_count", "1", "--repetitions", "1"]),
                         ("stress", [])]:
        destination = root / phase
        destination.mkdir(parents=True, exist_ok=True)
        os.link(args.fixture, destination / "fixture.json")
        command = [sys.executable, str(script), "--root", str(destination), "--parent", args.parent, *extra]
        manifest = {"command": command, "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                    "source_manifest": str(Path(args.parent) / "full-manifest.json")}
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (root / "phase.txt").write_text(phase)
        with (destination / "driver.log").open("w") as output:
            result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        (destination / "exit-code.txt").write_text(str(result.returncode))
        if result.returncode:
            raise RuntimeError(f"{phase} failed; inspect {destination / 'driver.log'}")
        verify(destination)
        print(f"{phase}: PASSED", flush=True)
    (root / "phase.txt").write_text("COMPLETE")


if __name__ == "__main__":
    main()
