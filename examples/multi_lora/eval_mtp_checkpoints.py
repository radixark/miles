"""Evaluate the snapshots emitted by run_mtp_training.py --spec-eval-interval.

Run on an evaluation GPU with access to the gateway's checkpoint root. The
evaluation script receives the result name, merged HF model path, and `on`.
It must run SGLang NEXTN and write <name>.json into SMOKE_EVAL_LOG_DIR with
metrics.average_accept_length, as the Phase 1 evaluator does. Rerun this
command while training to process newly committed snapshots.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.multi_lora.merge_mtp_checkpoint import merge_mtp_checkpoint
from miles.tinker.core.utils import parse_tinker_path, resolve_checkpoint_dir
from miles.utils.file_utils import atomic_write_text


def _evaluate_snapshot(args, step, checkpoint):
    name = f"step-{step:04d}"
    result_path = args.output_dir / f"{name}.json"
    if result_path.exists():
        if json.loads(result_path.read_text())["checkpoint"] != checkpoint:
            raise ValueError(f"{result_path} already contains a result for a different checkpoint")
        return

    model_id, kind, checkpoint_name = parse_tinker_path(checkpoint)
    if kind != "weights":
        raise ValueError(f"draft evaluation requires a training checkpoint, got {checkpoint}")
    checkpoint_dir = Path(resolve_checkpoint_dir(str(args.checkpoint_root), model_id, kind, checkpoint_name))
    with TemporaryDirectory(dir=args.output_dir) as directory:
        temporary = Path(directory)
        merged = temporary / "model"
        merge_started_at = time.perf_counter()
        merge_mtp_checkpoint(args.base_model, checkpoint_dir, merged)
        evaluation_started_at = time.perf_counter()
        subprocess.run(
            ["bash", str(args.eval_script), name, str(merged), "on"],
            env=os.environ | {"SMOKE_EVAL_LOG_DIR": str(temporary)},
            check=True,
        )
        finished_at = time.perf_counter()
        result = json.loads((temporary / f"{name}.json").read_text())
        if result["metrics"]["average_accept_length"] is None:
            raise ValueError(f"{checkpoint} had no speculative verification steps")
        result.update(
            step=step,
            checkpoint=checkpoint,
            merge_seconds=evaluation_started_at - merge_started_at,
            evaluation_process_seconds=finished_at - evaluation_started_at,
            total_seconds=finished_at - merge_started_at,
        )
        atomic_write_text(result_path, json.dumps(result, indent=2) + "\n")
        print(f"Saved {result_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("metrics", "checkpoint-root", "base-model", "eval-script", "output-dir"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.metrics.open() as metrics:
        for line in metrics:
            if not line.endswith("\n"):
                break  # A live training process may still be appending this line.
            record = json.loads(line)
            if checkpoint := record.get("spec_eval_checkpoint"):
                _evaluate_snapshot(args, record["step"], checkpoint)


if __name__ == "__main__":
    main()
