"""Check the trained samples a TMax run dumped against its episode traces.

    python examples/tmax/tools/check_samples.py <run dir>/rollout_data/*.pt --trace-dir <run dir>

Reads the batches `dump_training_data` writes (`rollout_data/<step>.pt`) and prints, per batch:

* samples, groups and samples per group, and sample status;
* length consistency: `loss_mask` and `rollout_log_probs` cover the response;
* the framing around each untrained (loss-mask 0) stretch, with the tool output replaced by <OBS>, so it can be
  compared with open-instruct's tool-response framing;
* how trained (loss-mask 1) stretches end, normally `<|im_end|>`;
* with --trace-dir, whether each sample's number of trained stretches equals the `turns` of a traced episode of the
  same task. `context_full` episodes may lose their last turn, so they are counted separately.

Runs anywhere torch and transformers are installed; set HF_HUB_OFFLINE=1 with a cached tokenizer.
"""

import argparse
import collections
import glob
import re

import torch
from transformers import AutoTokenizer

# A trace line starts {"time": ..., "task_id": ..., ..., "metrics": {...}, "messages": [...]}; reading the head of the
# line is enough and keeps multi-GB trace files fast.
_TRACE_HEAD = re.compile(rb'"task_id": "([^"]+)".*?"turns": (\d+).*?"end_reason": "([a-z_]+)"')
_TRACE_HEAD_BYTES = 8000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dumps", nargs="+", help="rollout_data/<step>.pt files")
    parser.add_argument("--trace-dir", help="the run's trace directory, holding traces-<host>.jsonl")
    parser.add_argument("--tokenizer", default="hamishivi/Qwen3.5-9B", help="tokenizer of the trained model")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    traces = _read_traces(args.trace_dir) if args.trace_dir else None
    for path in args.dumps:
        _check_dump(path, tokenizer, traces)


def _read_traces(trace_dir: str) -> dict[str, list[tuple[int, str]]]:
    """task_id -> [(turns, end_reason)] over every traced episode."""
    traces: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for path in glob.glob(f"{trace_dir}/traces-*.jsonl"):
        with open(path, "rb") as f:
            for line in f:
                if match := _TRACE_HEAD.search(line, 0, _TRACE_HEAD_BYTES):
                    task_id, turns, end_reason = match.groups()
                    traces[task_id.decode()].append((int(turns), end_reason.decode()))
    return traces


def _check_dump(path: str, tokenizer, traces: dict | None) -> None:
    samples = torch.load(path, weights_only=False)["samples"]
    groups = collections.Counter(s["group_index"] for s in samples)
    status, problems = collections.Counter(), collections.Counter()
    framings, trained_ends, turn_checks = collections.Counter(), collections.Counter(), collections.Counter()
    for sample in samples:
        status[str(sample["status"])] += 1
        response_length = sample["response_length"]
        mask = list(sample.get("loss_mask") or [])
        if len(mask) != response_length:
            problems["loss_mask length != response_length"] += 1
        if len(sample.get("rollout_log_probs") or []) != response_length:
            problems["rollout_log_probs length != response_length"] += 1
        response = sample["tokens"][len(sample["tokens"]) - response_length :]
        stretches = _stretches(mask)
        for value, start, end in stretches:
            text = tokenizer.decode(response[start:end])
            if value:
                trained_ends[repr(text[-12:])] += 1
            else:
                framings[_framing(text)[:200]] += 1
        if traces is not None:
            task_id = (sample.get("metadata") or {}).get("task_id")
            turn_checks[_turn_check(sum(1 for value, _, _ in stretches if value), traces.get(task_id, []))] += 1

    print(f"== {path}")
    print(
        f"   samples {len(samples)}; groups {len(groups)}; samples per group {dict(collections.Counter(groups.values()))}"
    )
    print(f"   status {dict(status)}")
    print(f"   length problems: {dict(problems) or 'none'}")
    print("   untrained-stretch framing (top 4):")
    for framing, count in framings.most_common(4):
        print(f"     {count:6d}  {framing!r}")
    print(f"   trained stretches end with (top 3): {trained_ends.most_common(3)}")
    if traces is not None:
        print(f"   vs traces: {dict(turn_checks)}")


def _stretches(mask: list[int]) -> list[tuple[int, int, int]]:
    """(value, start, end) runs of a 0/1 loss mask."""
    runs, start = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            runs.append((mask[start], start, i))
            start = i
    return runs


def _framing(text: str) -> str:
    """The text around a tool observation, with the observation replaced by <OBS>."""
    return re.sub(r"(<tool_response>\n?).*?(\n?</tool_response>)", r"\1<OBS>\2", text, flags=re.S)


def _turn_check(trained_turns: int, episodes: list[tuple[int, str]]) -> str:
    if not episodes:
        return "no trace of the task"
    if any(turns == trained_turns for turns, _ in episodes):
        return "turns match a trace"
    if any(end == "context_full" and turns == trained_turns + 1 for turns, end in episodes):
        return "context_full, last turn dropped"
    return "no trace with these turns"


if __name__ == "__main__":
    main()
