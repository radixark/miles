from pathlib import Path

from miles.utils.test_utils.comparisons.comparators import run_comparator

# Shared regexes for model-input / metadata tensors that are not weights or grads to
# compare. Exposed as named constants (not as defaults) so each test passes them
# explicitly -- every pass/fail knob is visible at the call site, nothing is implicit.
INPUT_TENSORS_SKIP_PATTERN: str = "input_ids|positions|cu_seqlens_q|cu_seqlens_kv|qkv_format|.*witness.*"
INPUT_TENSORS_ALLOW_FAILED_PATTERN: str = "input_ids|positions|cu_seqlens_q|cu_seqlens_kv|qkv_format"


def compare_dumps(
    baseline_dir: str,
    target_dir: str,
    *,
    diff_thresholds: list[tuple[str, str]],
    allow_skipped_pattern: str,
    allow_failed_pattern: str,
    phase_subdir: str | None = None,
    grouping_skip_keys: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> None:
    subdir = phase_subdir or ""
    baseline_root = Path(baseline_dir) / "dumps" / subdir
    target_root = Path(target_dir) / "dumps" / subdir

    assert baseline_root.exists(), f"Baseline dump dir does not exist: {baseline_root}"
    assert target_root.exists(), f"Target dump dir does not exist: {target_root}"

    # Dumps are segmented into leaf dirs (e.g. fwd_bwd/rollout_<id>), each a flat set of
    # .pt files with its own per-leaf step numbering. The sglang comparator compares one
    # flat dir at a time, so compare each matching leaf pair independently.
    baseline_leaves = _find_leaf_dump_dirs(baseline_root)
    target_leaves = _find_leaf_dump_dirs(target_root)

    assert baseline_leaves, f"No .pt dump files found under {baseline_root}"
    assert baseline_leaves == target_leaves, (
        f"Dump leaf-dir mismatch: baseline={baseline_leaves} vs target={target_leaves} "
        f"(under {baseline_root} vs {target_root})"
    )

    failed_leaves: list[str] = []
    for leaf in baseline_leaves:
        result = run_comparator(
            baseline_path=baseline_root / leaf,
            target_path=target_root / leaf,
            diff_thresholds=diff_thresholds,
            allow_skipped_pattern=allow_skipped_pattern,
            allow_failed_pattern=allow_failed_pattern,
            grouping_skip_keys=grouping_skip_keys,
            extra_args=extra_args,
        )
        if result.returncode != 0:
            failed_leaves.append(leaf)

    assert not failed_leaves, (
        f"Dump comparator failed (rc!=0) for {len(failed_leaves)}/{len(baseline_leaves)} leaf dir(s): "
        f"{failed_leaves} (baseline {baseline_root} vs target {target_root}). The comparator applies the "
        f"per-tensor predicates ({diff_thresholds}) and the allow/skip patterns itself; see "
        f"comparator_report.jsonl under {target_root}/<leaf> for the offending tensors."
    )
    print(f"Dump comparison passed: {len(baseline_leaves)} leaf dir(s) under {baseline_root} vs {target_root}")


def _find_leaf_dump_dirs(root: Path) -> list[str]:
    leaves: set[str] = {str(p.parent.relative_to(root)) for p in root.rglob("*.pt")}
    return sorted(leaves)
