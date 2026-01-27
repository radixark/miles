#!/usr/bin/env python3
"""
Script to consolidate consecutive duplicate timer entries in miles timer logs.
Handles multiple types of gather operations by summing their latencies:
- non_expert_all_tp_gather_source*
- expert_all_gather_name_param_tp_gather* and expert_all_gather_name_param_ep_gather_source*
  (these two are treated as the same group and squashed together)
"""

import re
import sys


def parse_timer_line(line: str) -> tuple[str, float]:
    """Parse a timer log line and extract the timer name and elapsed time."""
    # Pattern: Timer <name> end (elapsed: <time>ms)
    match = re.match(r"Timer (.+?) end \(elapsed: ([\d.]+)ms\)", line.strip())
    if match:
        timer_name = match.group(1)
        elapsed_time = float(match.group(2))
        return timer_name, elapsed_time
    else:
        raise ValueError(f"Unable to parse timer line: {line}")


def get_timer_group(timer_name: str) -> str | None:
    """
    Get the consolidation group for a timer name.
    Returns the group name if the timer should be consolidated, None otherwise.
    Timers in the same group will be squashed together.
    """
    # Group 1: non_expert_all_tp_gather_source (standalone group)
    if timer_name.startswith("non_expert_all_tp_gather_source"):
        return "non_expert_all_tp_gather_source"

    # Group 2: expert param gather operations (tp_gather and ep_gather treated as same)
    expert_gather_patterns = [
        "expert_all_gather_name_param_tp_gather",
        "expert_all_gather_name_param_ep_gather_source",
    ]
    if any(timer_name.startswith(pattern) for pattern in expert_gather_patterns):
        return "expert_all_gather_name_param_gather"

    return None


def is_consolidatable_timer(timer_name: str) -> bool:
    """
    Check if a timer name should be consolidated with consecutive identical entries.
    Returns True for gather operation timers that tend to have many consecutive entries.
    """
    return get_timer_group(timer_name) is not None


def format_timer_line(timer_name: str, elapsed_time: float) -> str:
    """Format a timer line with the given name and elapsed time."""
    return f"Timer {timer_name} end (elapsed: {elapsed_time:.3f}ms)"


def consolidate_timer_log(input_file: str, output_file: str) -> None:
    """
    Consolidate consecutive duplicate timer entries for gather operations
    by summing their latencies, while preserving the order of other entries.
    """

    with open(input_file) as f:
        lines = f.readlines()

    consolidated_lines = []
    current_group_name = None
    current_group_total = 0.0
    current_group_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            timer_name, elapsed_time = parse_timer_line(line)
        except ValueError:
            # If we can't parse the line, just keep it as is
            if current_group_name:
                # Flush any pending group
                consolidated_lines.append(
                    format_timer_line(current_group_name, current_group_total)
                    + f" (consolidated {current_group_count} entries)"
                )
                current_group_name = None
                current_group_total = 0.0
                current_group_count = 0
            consolidated_lines.append(line)
            continue

        # Check if this is a consolidatable timer entry
        timer_group = get_timer_group(timer_name)

        if timer_group:
            if current_group_name == timer_group:
                # Continue accumulating the same group
                current_group_total += elapsed_time
                current_group_count += 1
            else:
                # Flush previous group if exists
                if current_group_name:
                    consolidated_lines.append(
                        format_timer_line(current_group_name, current_group_total)
                        + f" (consolidated {current_group_count} entries)"
                    )

                # Start new group
                current_group_name = timer_group
                current_group_total = elapsed_time
                current_group_count = 1
        else:
            # Different timer - flush any pending group and add this line
            if current_group_name:
                consolidated_lines.append(
                    format_timer_line(current_group_name, current_group_total)
                    + f" (consolidated {current_group_count} entries)"
                )
                current_group_name = None
                current_group_total = 0.0
                current_group_count = 0

            # Add non-target timer as-is
            consolidated_lines.append(format_timer_line(timer_name, elapsed_time))

    # Flush any remaining group
    if current_group_name:
        consolidated_lines.append(
            format_timer_line(current_group_name, current_group_total)
            + f" (consolidated {current_group_count} entries)"
        )

    # Write consolidated output
    with open(output_file, "w") as f:
        for line in consolidated_lines:
            f.write(line + "\n")

    print("Consolidation complete!")
    print(f"Input: {input_file} ({len(lines)} lines)")
    print(f"Output: {output_file} ({len(consolidated_lines)} lines)")


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
        sys.exit(1)


if __name__ == "__main__":
    main()
