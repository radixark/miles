"""Compare --rematerialize-param-from-master-weight off vs on from two run logs.

usage: python exp_1572_analyze.py <off.log> <on.log>
"""

import re
import statistics
import sys
from pathlib import Path

CPU_MEM = re.compile(r"\[CPU memory\] (\w+): ([\d.]+) GB \(rollout_id=(\d+)")
GPU_MEM = re.compile(r"Memory-Usage ([\w ]+): \{'gpu': '(\d+)'.*?'used_GB': ([\d.]+).*?'reserved_GB': ([\d.]+)\}")
TIMER = re.compile(r"Timer (\w+) end \(elapsed: ([\d.]+)s\)")


def parse(path):
    text = Path(path).read_text(errors="replace")
    cpu = {}  # label -> {rollout: gb}
    for label, gb, rid in CPU_MEM.findall(text):
        cpu.setdefault(label, {})[int(rid)] = float(gb)
    gpu = {}  # phase -> list of used_GB (all ranks)
    for phase, _rank, used, reserved in GPU_MEM.findall(text):
        gpu.setdefault(phase.strip(), []).append((float(used), float(reserved)))
    timers = {}
    for name, secs in TIMER.findall(text):
        timers.setdefault(name, []).append(float(secs))
    return cpu, gpu, timers


def fmt_delta(a, b, unit, lower_is_better=True):
    if a is None or b is None:
        return "n/a"
    d = b - a
    pct = (d / a * 100) if a else 0.0
    mark = "" if (d < 0) == lower_is_better or abs(pct) < 0.5 else " ⚠"
    return f"{a:.2f} → {b:.2f} {unit}  ({d:+.2f}, {pct:+.1f}%){mark}"


def main(off_path, on_path):
    off_cpu, off_gpu, off_t = parse(off_path)
    on_cpu, on_gpu, on_t = parse(on_path)

    print("=" * 78)
    print("HOST MEMORY  (psutil virtual_memory().used, whole node)")
    print("=" * 78)
    for label in sorted(set(off_cpu) | set(on_cpu)):
        a, b = off_cpu.get(label, {}), on_cpu.get(label, {})
        for rid in sorted(set(a) | set(b)):
            print(f"  {label} rollout {rid}: {fmt_delta(a.get(rid), b.get(rid), 'GB')}")
        if a and b:
            print(f"  {label} MEAN     : {fmt_delta(statistics.mean(a.values()), statistics.mean(b.values()), 'GB')}")

    print()
    print("=" * 78)
    print("GPU MEMORY  (max used_GB / max reserved_GB across ranks+samples)")
    print("=" * 78)
    for phase in sorted(set(off_gpu) | set(on_gpu)):
        a, b = off_gpu.get(phase, []), on_gpu.get(phase, [])
        au = max((u for u, _ in a), default=None)
        bu = max((u for u, _ in b), default=None)
        ar = max((r for _, r in a), default=None)
        br = max((r for _, r in b), default=None)
        # 'before/after update_weights' is where the flag keeps the param buffer resident
        print(f"  {phase:28s} used     {fmt_delta(au, bu, 'GB', lower_is_better=True)}")
        print(f"  {'':28s} reserved {fmt_delta(ar, br, 'GB', lower_is_better=True)}")

    print()
    print("=" * 78)
    print("TIMERS  (mean over occurrences; n shown)")
    print("=" * 78)
    for name in sorted(set(off_t) | set(on_t)):
        a, b = off_t.get(name, []), on_t.get(name, [])
        am = statistics.mean(a) if a else None
        bm = statistics.mean(b) if b else None
        print(f"  {name:24s} n={len(a)}/{len(b)}  {fmt_delta(am, bm, 's')}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
