"""
Efficient bulk serialization utilities for database I/O
"""

import json
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)


def save_programs_bulk(programs: Dict[str, Any], save_path: str, prompts_by_program: Dict[str, dict] = None) -> None:
    """
    Save all programs to a single JSON file for efficient I/O

    Args:
        programs: Dictionary of {program_id: Program} objects
        save_path: Base directory path
        prompts_by_program: Optional prompts dictionary

    File structure:
        {save_path}/programs_bulk.json
    """
    os.makedirs(save_path, exist_ok=True)

    # Serialize all programs
    programs_data = {}
    for program_id, program in programs.items():
        program_dict = program.to_dict()

        # Add prompts if available
        if prompts_by_program and program_id in prompts_by_program:
            program_dict["prompts"] = prompts_by_program[program_id]

        programs_data[program_id] = program_dict

    # Write to single file
    bulk_file_path = os.path.join(save_path, "programs_bulk.json")
    with open(bulk_file_path, "w") as f:
        json.dump(programs_data, f, separators=(',', ':'))  # Compact format

    logger.info(f"Saved {len(programs_data)} programs to bulk file: {bulk_file_path}")


def load_programs_bulk(save_path: str) -> Dict[str, dict]:
    """
    Load all programs from a single JSON file

    Args:
        save_path: Base directory path

    Returns:
        Dictionary of {program_id: program_dict}
    """
    bulk_file_path = os.path.join(save_path, "programs_bulk.json")

    if not os.path.exists(bulk_file_path):
        logger.warning(f"Bulk file not found: {bulk_file_path}")
        return {}

    with open(bulk_file_path, "r") as f:
        programs_data = json.load(f)

    logger.info(f"Loaded {len(programs_data)} programs from bulk file: {bulk_file_path}")
    return programs_data


def save_programs_individual(programs: Dict[str, Any], save_path: str, prompts_by_program: Dict[str, dict] = None) -> None:
    """
    Save programs as individual files (legacy mode)

    NOTE: This function is currently NOT used in the main save() flow.
    Legacy mode directly uses the existing _save_program() method in a loop.
    This function is kept for potential future use cases (e.g., incremental saves,
    external tools, or alternative save strategies).

    Args:
        programs: Dictionary of {program_id: Program} objects
        save_path: Base directory path
        prompts_by_program: Optional prompts dictionary

    File structure:
        {save_path}/programs/{program_id}.json
    """
    programs_dir = os.path.join(save_path, "programs")
    os.makedirs(programs_dir, exist_ok=True)

    for program_id, program in programs.items():
        program_dict = program.to_dict()

        if prompts_by_program and program_id in prompts_by_program:
            program_dict["prompts"] = prompts_by_program[program_id]

        program_path = os.path.join(programs_dir, f"{program_id}.json")
        with open(program_path, "w") as f:
            json.dump(program_dict, f)

    logger.info(f"Saved {len(programs)} programs to individual files in {programs_dir}")


def load_programs_individual(save_path: str) -> Dict[str, dict]:
    """
    Load programs from individual files (legacy mode compatibility)

    This function is used to load old checkpoints that were saved with
    the legacy individual-file format (programs/*.json).

    Args:
        save_path: Base directory path

    Returns:
        Dictionary of {program_id: program_dict}
    """
    programs_dir = os.path.join(save_path, "programs")
    programs_data = {}

    if not os.path.exists(programs_dir):
        logger.warning(f"Programs directory not found: {programs_dir}")
        return {}

    for program_file in os.listdir(programs_dir):
        if program_file.endswith(".json"):
            program_path = os.path.join(programs_dir, program_file)
            with open(program_path, "r") as f:
                program_dict = json.load(f)

            program_id = program_dict.get("id")
            if program_id:
                programs_data[program_id] = program_dict
            else:
                logger.warning(f"Program file missing 'id' field: {program_file}")

    logger.info(f"Loaded {len(programs_data)} programs from individual files in {programs_dir}")
    return programs_data
