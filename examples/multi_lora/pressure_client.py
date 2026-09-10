"""Launch N independent Tinker E2E users against an already running gateway.

N comes from slot_capacity.py's report, or an explicit --clients count for a
later capacity sweep. Each task runs the exact same single-user SDK script.
Only model creation and final completion have gates; training steps are independent.

Example:
    python -m examples.multi_lora.pressure_client --capacity-report /shared/capacity.json \
        --model /models/Qwen3-30B-A3B --dataset /datasets/dapo-math-17k.jsonl \
        --output-dir /shared/n-users
"""

import asyncio
import json
import traceback
from pathlib import Path

from examples.multi_lora.pressure_dapo import _write_json
from examples.multi_lora.tinker_e2e_user import _parser, _prepare, run_user


async def _main(args):
    if args.capacity_report is not None:
        report = json.loads(args.capacity_report.read_text())
        args.clients = report["n_slots"]
        if report.get("max_tokens_per_gpu") != 8192:
            raise ValueError("capacity report must use the same 8192-token workload")
        if report.get("lora_rank") != args.lora_rank or report.get("hf_checkpoint") != args.model:
            raise ValueError("capacity report model/rank must match the SDK workload")
    if type(args.clients) is not int or args.clients < 1:
        raise ValueError("N must be a positive integer")
    examples, tokenizer = _prepare(args)
    ready, finished = asyncio.Barrier(args.clients), asyncio.Barrier(args.clients)
    try:
        async with asyncio.timeout(args.timeout_seconds), asyncio.TaskGroup() as tasks:
            running = [
                tasks.create_task(run_user(i, args, examples, tokenizer, ready=ready, finished=finished))
                for i in range(args.clients)
            ]
        results = [task.result() for task in running]
        assert len({result["model_id"] for result in results}) == args.clients
        assert all(result["steps"] == args.steps and args.steps > 0 for result in results)
        timings = [
            json.loads((args.output_dir / f"{result['adapter']}-timing-summary.json").read_text())
            for result in results
        ]
        _write_json(args.output_dir / "timing-comparison.json", {"users": timings})
        _write_json(args.output_dir / "result.json", {"status": "passed", "n": args.clients, "users": results})
    except BaseException:
        # TaskGroup cancels peers, including tasks at either gate. Timeouts,
        # filtering exhaustion and RPC failures are not relabelled OOM.
        _write_json(
            args.output_dir / "result.json", {"status": "failed", "n": args.clients, "error": traceback.format_exc()}
        )
        raise


if __name__ == "__main__":
    parser = _parser()
    parser.description = __doc__
    count = parser.add_mutually_exclusive_group(required=True)
    count.add_argument("--capacity-report", type=Path)
    count.add_argument("--clients", type=int, help="Explicit N; the gateway must already have N slots")
    asyncio.run(_main(parser.parse_args()))
