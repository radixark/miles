import argparse
import json
import os
import subprocess
import time

import torch

from tests.ci.ci_register import HWBackend, collect_tests
from tests.ci.ci_utils import reaping_is_isolated, run_unittest_files

parser = argparse.ArgumentParser()
selector = parser.add_mutually_exclusive_group(required=True)
selector.add_argument("--test", choices=("all", "pool", "sft", "lora", "mhc", "fake_quant"))
selector.add_argument("--file")
parser.add_argument("--gpus", type=int, required=True)
args = parser.parse_args()
files = {
    "sft": "tests/e2e/short/test_qwen3_0.6B_sft_snapshot_eval.py",
    "lora": "tests/e2e/lora/test_lora_qwen2.5_0.5B.py",
}
if args.test != "all":
    files.update(mhc="tests/fast-gpu/kernels/hyper_connection/test_mhc.py", fake_quant="tests/fast-gpu/kernels/quant/test_fake_quant.py")
selected = [args.file] if args.file else (list(files.values()) if args.test in ("all", "pool") else [files[args.test]])
tests = [test for test in collect_tests(selected, sanity_check=True) if test.backend == HWBackend.CUDA]
assert len(tests) == len(selected)
assert torch.cuda.device_count() == args.gpus
assert reaping_is_isolated()
record = {
    "files": selected,
    "gpus": args.gpus,
    "cvd": os.environ["CUDA_VISIBLE_DEVICES"],
    "gpu_uuids": [str(torch.cuda.get_device_properties(i).uuid) for i in range(args.gpus)],
    "miles_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "started_at": time.time(),
}
print("BENCHMARK_START " + json.dumps(record), flush=True)
result = run_unittest_files(tests, timeout_per_file=1800, reap_leftovers=True)
record.update(finished_at=time.time(), exit_code=result)
record["elapsed_s"] = record["finished_at"] - record["started_at"]
print("BENCHMARK_RESULT " + json.dumps(record), flush=True)
raise SystemExit(result)
