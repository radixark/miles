import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from miles.utils.debug_utils.run_megatron.logprob_comparator import compare_logprobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run same-token SGLang/Megatron prefill parity.")
    parser.add_argument("--sglang-checkpoint", type=Path, required=True)
    parser.add_argument("--megatron-hf-checkpoint", type=Path, required=True)
    parser.add_argument("--megatron-checkpoint", type=Path, required=True)
    parser.add_argument("--token-ids-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-type", default="kimi-k3-4layer")
    parser.add_argument("--sglang-tp", type=int, default=1)
    parser.add_argument("--megatron-tp", type=int, default=1)
    parser.add_argument("--megatron-ep", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    token_ids = json.loads(args.token_ids_file.read_text())
    assert isinstance(token_ids, list) and len(token_ids) >= 2
    assert all(type(token_id) is int for token_id in token_ids)

    sglang_output = args.output_dir / "sglang" / "logprobs"
    megatron_output = args.output_dir / "megatron"
    shutil.rmtree(args.output_dir, ignore_errors=True)

    run(
        [
            sys.executable,
            str(Path(__file__).with_name("run_sglang_prefill.py")),
            "--model-path",
            str(args.sglang_checkpoint),
            "--token-ids-file",
            str(args.token_ids_file),
            "--output-dir",
            str(sglang_output),
            "--tp-size",
            str(args.sglang_tp),
            "--mem-fraction-static",
            str(args.mem_fraction_static),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "miles.utils.debug_utils.run_megatron",
            "run",
            "--model-type",
            args.model_type,
            "--hf-checkpoint",
            str(args.megatron_hf_checkpoint),
            "--ref-load",
            str(args.megatron_checkpoint),
            "--token-ids-file",
            str(args.token_ids_file),
            "--seq-length",
            str(len(token_ids)),
            "--tp",
            str(args.megatron_tp),
            "--ep",
            str(args.megatron_ep),
            "--output-dir",
            str(megatron_output / "dumps"),
            "--logprob-output",
            str(megatron_output / "logprobs"),
        ]
    )

    passed = compare_logprobs(
        baseline_dir=sglang_output,
        target_dir=megatron_output / "logprobs",
        threshold=args.threshold,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
