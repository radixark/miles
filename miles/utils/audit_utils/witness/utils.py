from collections.abc import Iterable, Sequence


def compute_id_ranges(ids: Iterable[int]) -> list[tuple[int, int]]:
    runs: list[list[int]] = []
    for x in ids:
        if runs and runs[-1][1] == x:
            runs[-1][1] += 1
        else:
            runs.append([x, x + 1])

    merged: list[tuple[int, int]] = []
    for start, stop in sorted(runs):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def exclude_id_ranges(ids: Iterable[int], ranges: Sequence[tuple[int, int]]) -> set[int]:
    result = set(ids)
    for start, stop in ranges:
        if stop - start <= len(result):
            result.difference_update(range(start, stop))
        else:
            result = {x for x in result if not start <= x < stop}
    return result
