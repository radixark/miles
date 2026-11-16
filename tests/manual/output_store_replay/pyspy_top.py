"""Inclusive and leaf sample shares per frame of py-spy raw (collapsed) stacks.

    python tests/manual/output_store_replay/pyspy_top.py <pyspy.txt>... [--top N] [--under SUBSTRING]

Frames read `function (file:line)`; --under keeps only the samples whose stack has a frame containing SUBSTRING.
"""

import collections
import sys


def main(argv: list[str]) -> None:
    top, under, paths = 25, None, []
    it = iter(argv)
    for arg in it:
        if arg == "--top":
            top = int(next(it))
        elif arg == "--under":
            under = next(it)
        else:
            paths.append(arg)
    inclusive, leaf, total = collections.Counter(), collections.Counter(), 0
    for path in paths:
        for line in open(path):
            if not line.strip():
                continue
            stack, _, count = line.rstrip("\n").rpartition(" ")
            # A thread sampled in native code can have no Python frame.
            frames = [frame for frame in stack.split(";") if frame] or ["(no Python frame)"]
            if under and not any(under in frame for frame in frames):
                continue
            total += int(count)
            for frame in set(frames):
                inclusive[frame] += int(count)
            leaf[frames[-1]] += int(count)
    print(f"samples {total}")
    for name, counter in (("inclusive", inclusive), ("leaf", leaf)):
        print(f"--- {name}")
        for frame, count in counter.most_common(top):
            print(f"{100 * count / max(total, 1):5.1f}%  {frame[:160]}")


if __name__ == "__main__":
    main(sys.argv[1:])
