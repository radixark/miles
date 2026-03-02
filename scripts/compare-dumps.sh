#!/bin/bash
#
# Compare tensor dumps between baseline and target runs.
#
# Usage:
#   bash scripts/compare-dumps.sh baseline target
#   bash scripts/compare-dumps.sh baseline target --phase fwd_bwd
#   bash scripts/compare-dumps.sh baseline target --phase fwd_only --filter "hidden_states"
#   bash scripts/compare-dumps.sh baseline target --diff-threshold 1e-4
#
# Environment:
#   DUMPER_BASE_DIR=/tmp/dumper  (default, override to match dump-small-next.sh)
#

set -euo pipefail

BASELINE_TAG="${1:?Usage: $0 <baseline_tag> <target_tag> [options]}"
TARGET_TAG="${2:?Usage: $0 <baseline_tag> <target_tag> [options]}"
shift 2

DUMPER_BASE_DIR="${DUMPER_BASE_DIR:-/tmp/dumper}"
BASELINE_DIR="${DUMPER_BASE_DIR}/${BASELINE_TAG}"
TARGET_DIR="${DUMPER_BASE_DIR}/${TARGET_TAG}"

# Parse optional args
PHASE=""
FILTER=""
DIFF_THRESHOLD="1e-3"

while [[ $# -gt 0 ]]; do
   case $1 in
      --phase) PHASE="$2"; shift 2 ;;
      --filter) FILTER="$2"; shift 2 ;;
      --diff-threshold) DIFF_THRESHOLD="$2"; shift 2 ;;
      *) echo "Unknown arg: $1"; exit 1 ;;
   esac
done

# Determine which phases to compare
if [ -n "$PHASE" ]; then
   PHASES=("$PHASE")
else
   PHASES=()
   for p in inference fwd_only fwd_bwd; do
      if [ -d "${TARGET_DIR}/${p}" ] && [ -d "${BASELINE_DIR}/${p}" ]; then
         PHASES+=("$p")
      fi
   done
fi

if [ ${#PHASES[@]} -eq 0 ]; then
   echo "Error: no matching phases found between ${BASELINE_DIR} and ${TARGET_DIR}"
   echo ""
   echo "Baseline contents:"
   ls -la "${BASELINE_DIR}/" 2>/dev/null || echo "  (not found)"
   echo "Target contents:"
   ls -la "${TARGET_DIR}/" 2>/dev/null || echo "  (not found)"
   exit 1
fi

echo "=== Dump Comparison ==="
echo "Baseline: ${BASELINE_DIR}"
echo "Target:   ${TARGET_DIR}"
echo "Phases:   ${PHASES[*]}"
echo "Threshold: ${DIFF_THRESHOLD}"
[ -n "$FILTER" ] && echo "Filter:   ${FILTER}"
echo ""

# Find the actual sglang_dump_* subdirectory inside each phase dir
find_dump_subdir() {
   local phase_dir="$1"
   # The dumper creates sglang_dump_<timestamp>/ inside the phase dir,
   # or dumps directly into the phase dir depending on configuration.
   # Check for sglang_dump_* subdirs first.
   local subdir
   subdir=$(find "$phase_dir" -maxdepth 1 -type d -name 'sglang_dump_*' | head -1)
   if [ -n "$subdir" ]; then
      echo "$subdir"
   else
      # Dumps might be directly in the phase dir
      echo "$phase_dir"
   fi
}

EXIT_CODE=0

for phase in "${PHASES[@]}"; do
   echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
   echo "Phase: ${phase}"
   echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

   baseline_phase_dir=$(find_dump_subdir "${BASELINE_DIR}/${phase}")
   target_phase_dir=$(find_dump_subdir "${TARGET_DIR}/${phase}")

   baseline_count=$(find "$baseline_phase_dir" -name '*.pt' 2>/dev/null | wc -l)
   target_count=$(find "$target_phase_dir" -name '*.pt' 2>/dev/null | wc -l)
   echo "Baseline: ${baseline_count} tensors in ${baseline_phase_dir}"
   echo "Target:   ${target_count} tensors in ${target_phase_dir}"
   echo ""

   COMPARE_ARGS=(
      --baseline-path "$baseline_phase_dir"
      --target-path "$target_phase_dir"
      --diff-threshold "$DIFF_THRESHOLD"
   )
   [ -n "$FILTER" ] && COMPARE_ARGS+=(--filter "$FILTER")

   python3 -m sglang.srt.debug_utils.dump_comparator "${COMPARE_ARGS[@]}" || EXIT_CODE=1

   echo ""
done

if [ $EXIT_CODE -eq 0 ]; then
   echo "=== All phases compared successfully ==="
else
   echo "=== Some comparisons had errors (exit code ${EXIT_CODE}) ==="
fi

exit $EXIT_CODE
