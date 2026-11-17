"""Per-step perf times and the train/rollout log-prob difference from one run's log, as a table.

    python tests/manual/output_store_replay/step_metrics.py <run.log>...

Reads the metric dicts Miles prints; only numbers are printed, since a raw log may hold the W&B key.
"""

import collections
import re
import sys

KEYS = (
    "perf/step_time",
    "perf/rollout_time",
    "perf/data_preprocess_time",
    "perf/replay_fill_time",
    "perf/actor_train_time",
    "perf/train_time",
    "train/train_rollout_logprob_abs_diff",
)
PAIR = re.compile(r"'([A-Za-z0-9_/]+)': (-?[0-9.]+(?:e-?[0-9]+)?)")


def per_step(path: str) -> dict[str, list[float]]:
    values = collections.defaultdict(list)
    for line in open(path, errors="replace"):
        for key, value in PAIR.findall(line):
            # A metric dict can be printed twice in a row.
            if key in KEYS and (not values[key] or values[key][-1] != float(value)):
                values[key].append(float(value))
    return values


if __name__ == "__main__":
    for log in sys.argv[1:]:
        values = per_step(log)
        print(log)
        for key in KEYS:
            print(f"  {key:<40}" + "".join(f"{v:>10.4g}" for v in values[key]))
