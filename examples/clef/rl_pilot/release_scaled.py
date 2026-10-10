"""Finalize and publish a fully reviewed scalable workflow dataset."""
import json
import os
import subprocess
import time
from pathlib import Path

from tap import Tap


class Args(Tap):
    root: Path
    python: Path
    prefix: str = "shi/decision/clef-rl-scaled-32768-v1"


def main(args: Args) -> None:
    exit_path = args.root / "generation.exit"
    while not exit_path.is_file():
        time.sleep(5)
    code = args.root / "code"
    env = dict(os.environ, PYTHONPATH=str(code))
    env.pop("AWS_PROFILE", None)
    success = int(exit_path.read_text()) == 0
    for retry in range(2):
        if success:
            break
        with (args.root / f"review-retry-{retry}.log").open("w") as log:
            result = subprocess.run(
                [str(args.python), "-u", "-m", "examples.clef.rl_pilot.scaled",
                 "--output", str(args.root / "data"), "--audit-effort", "high", "--audit-tokens", "12000"],
                env=env, cwd=code, stdout=log, stderr=subprocess.STDOUT,
            )
        success = result.returncode == 0
    if not success:
        raise RuntimeError("Generation/review incomplete; publication blocked")
    manifest = json.loads((args.root / "data/manifest.json").read_text())
    if manifest["train"] != 32768 or manifest["validation"] != 2048 or manifest["model"] != "gpt-6-luna":
        raise ValueError("Unexpected or unreviewed release")
    subprocess.run(
        [str(args.python), "-u", "-m", "examples.clef.rl_pilot.publish",
         "--data", str(args.root / "data"), "--prefix", args.prefix],
        env=env, cwd=code, check=True,
    )


if __name__ == "__main__":
    main(Args(underscores_to_dashes=True).parse_args())

