#!/usr/bin/env python3
"""
Script to consolidate miles timer logs into a concise per-step summary.

For each training step (delimited by 'update_weights_implementation end'),
accumulates all entries of each timer group and outputs one summary line per group.

Timer groups consolidated:
- non_expert_all_tp_gather_source* → non_expert_tp_gather
- expert_all_gather_name_param_tp_gather* → expert_tp_gather
- expert_all_gather_name_param_ep_gather_source* → expert_ep_gather
- load_weights_to_cpu_replica → load_weights_to_cpu_replica
- rdma_submit → rdma_submit
- expert_convert_to_hf → expert_convert_to_hf

Step-level timers preserved as-is:
- non_expert_transfer, expert_transfer, final_trans, update_weights_implementation
- rdma_move_replica_to_cpu, rdma_cpu_registration, on_transfer_start
"""

import re
import sys
from collections import defaultdict


# Step-level timers that should be printed as-is (not consolidated)
STEP_LEVEL_TIMERS = {
    "non_expert_transfer",
    "expert_transfer",
    "final_trans",
    "update_weights_implementation",
    "rdma_move_replica_to_cpu",
    "rdma_cpu_registration",
    "on_transfer_start",
}

# The timer that marks end of a step
STEP_DELIMITER = "update_weights_implementation"


def parse_timer_line(line: str) -> tuple[str, float, str] | None:
    """Parse a timer log line and extract the timer name, elapsed time, and optional metadata.

    Returns:
        (timer_name, elapsed_time, metadata_str) or None if the line is not a timer line.
    """
    match = re.match(r"Timer (.+?) end \(elapsed: ([\d.]+)ms\)(.*)$", line.strip())
    if match:
        timer_name = match.group(1)
        elapsed_time = float(match.group(2))
        metadata = match.group(3).strip()
        return timer_name, elapsed_time, metadata
    return None


def get_timer_group(timer_name: str) -> str | None:
    """
    Get the consolidation group for a timer name.
    Returns the group name if the timer should be consolidated, None otherwise.
    """
    if timer_name.startswith("non_expert_all_tp_gather_source"):
        return "non_expert_tp_gather"
    if timer_name.startswith("expert_all_gather_name_param_tp_gather"):
        return "expert_tp_gather"
    if timer_name.startswith("expert_all_gather_name_param_ep_gather_source"):
        return "expert_ep_gather"
    if timer_name == "load_weights_to_cpu_replica":
        return "load_weights_to_cpu_replica"
    if timer_name == "get_transfer_ready_params":
        return "get_transfer_ready_params"
    if timer_name == "rdma_submit":
        return "rdma_submit"
    if timer_name == "expert_convert_to_hf":
        return "expert_convert_to_hf"
    if timer_name == "rdma_sync_write":
        return "rdma_sync_write"
    if timer_name == "rdma_async_write":
        return "rdma_async_write"
    return None


def parse_experiment_blocks(lines: list[str]) -> tuple[dict | None, int]:
    """Parse EXPERIMENT START blocks from the beginning of the log.

    Returns the last experiment config dict and the line index where timer entries begin.
    """
    last_config = None
    current_config = None
    in_block = False
    timer_start_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("=" * 10):
            if in_block:
                # End of block
                in_block = False
                last_config = current_config
                timer_start_idx = i + 1
            else:
                # Start of block
                in_block = True
                current_config = {}
            continue

        if in_block:
            if stripped.startswith("EXPERIMENT START:"):
                if current_config is not None:
                    current_config["_timestamp"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Node rank:"):
                if current_config is not None:
                    current_config["_node_info"] = stripped
            elif stripped.startswith("Configuration:"):
                continue
            elif ":" in stripped:
                key, _, value = stripped.partition(":")
                if current_config is not None:
                    current_config[key.strip()] = value.strip()
            continue

        # First non-block, non-empty line that's not a separator → timer entries start
        if parse_timer_line(stripped) is not None or stripped.startswith("Timer"):
            timer_start_idx = i
            break

    return last_config, timer_start_idx


def consolidate_timer_log(input_file: str, output_file: str) -> None:
    """
    Consolidate timer log into a concise per-step summary.
    """
    with open(input_file) as f:
        lines = f.readlines()

    # Parse experiment config (keep only the last one)
    last_config, timer_start_idx = parse_experiment_blocks(lines)

    output_lines = []

    # Write single experiment header
    if last_config:
        output_lines.append("=" * 80)
        ts = last_config.pop("_timestamp", "unknown")
        node_info = last_config.pop("_node_info", "")
        output_lines.append(f"EXPERIMENT START: {ts}")
        if node_info:
            output_lines.append(node_info)
        output_lines.append("Configuration:")
        for key in sorted(last_config):
            output_lines.append(f"  {key}: {last_config[key]}")
        output_lines.append("=" * 80)
        output_lines.append("")

    # Per-step accumulation
    step_num = 0
    # group_name -> (total_ms, count)
    group_accum: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    # Step-level timers in order of appearance
    step_level_entries: list[str] = []
    metadata_str = ""

    # Consolidation group output order
    GROUP_ORDER = [
        "non_expert_tp_gather",
        "load_weights_to_cpu_replica",
        "rdma_submit",
        "get_transfer_ready_params",
        "expert_tp_gather",
        "expert_ep_gather",
        "expert_convert_to_hf",
    ]

    def flush_step():
        nonlocal group_accum, step_level_entries, step_num
        if not group_accum and not step_level_entries:
            return

        output_lines.append(f"--- Step {step_num} ---")

        # Print consolidated groups in defined order
        printed_groups = set()
        for group_name in GROUP_ORDER:
            if group_name in group_accum:
                total_ms, count = group_accum[group_name]
                output_lines.append(f"  {group_name}: {total_ms:.3f}ms total ({count} calls)")
                printed_groups.add(group_name)

        # Print any remaining groups not in ORDER
        for group_name in sorted(group_accum):
            if group_name not in printed_groups:
                total_ms, count = group_accum[group_name]
                output_lines.append(f"  {group_name}: {total_ms:.3f}ms total ({count} calls)")

        # Print step-level timers
        for entry in step_level_entries:
            output_lines.append(f"  {entry}")

        output_lines.append("")
        group_accum = defaultdict(lambda: (0.0, 0))
        step_level_entries = []
        step_num += 1

    for line in lines[timer_start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue

        # Skip any stray experiment block lines
        if stripped.startswith("=" * 10):
            continue
        if stripped.startswith("EXPERIMENT START:"):
            continue

        parsed = parse_timer_line(stripped)
        if parsed is None:
            # Non-timer line (config lines, etc.) — skip
            continue

        timer_name, elapsed_time, metadata = parsed
        if not metadata_str and metadata:
            metadata_str = metadata

        # Check if it's a step-level timer
        if timer_name in STEP_LEVEL_TIMERS:
            step_level_entries.append(f"{timer_name}: {elapsed_time:.3f}ms")
            if timer_name == STEP_DELIMITER:
                flush_step()
            continue

        # Check if it's a consolidatable group
        group = get_timer_group(timer_name)
        if group:
            prev_total, prev_count = group_accum[group]
            group_accum[group] = (prev_total + elapsed_time, prev_count + 1)
        else:
            # Unknown timer — add as step-level
            step_level_entries.append(f"{timer_name}: {elapsed_time:.3f}ms")

    # Flush any remaining entries (incomplete step)
    if group_accum or step_level_entries:
        flush_step()

    # Write output
    with open(output_file, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

    print("Consolidation complete!")
    print(f"Input: {input_file} ({len(lines)} lines)")
    print(f"Output: {output_file} ({len(output_lines)} lines)")


# Basic usage - creates miles_timer_0_consolidated.log
# python consolidate_timer_log.py miles_timer_0.log
def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) not in [2, 3]:
        print("Usage: python consolidate_timer_log.py <input_file> [output_file]")
        print("If output_file is not specified, will use input_file with '_consolidated' suffix")
        sys.exit(1)

    input_file = sys.argv[1]

    if len(sys.argv) == 3:
        output_file = sys.argv[2]
    else:
        # Generate output filename
        if input_file.endswith(".log"):
            output_file = input_file[:-4] + "_consolidated.log"
        else:
            output_file = input_file + "_consolidated"

    try:
        consolidate_timer_log(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        raise


if __name__ == "__main__":
    main()
