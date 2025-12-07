#!/usr/bin/env python3
"""
Extract prompts from OpenEvolve checkpoint JSON files and save them as text files.

This script traverses the openevolve_output_output directory structure,
reads JSON files from each checkpoint's programs directory,
and extracts the prompts data to organized text files.

python extract_prompts.py examples/go_ramsey_search/openevolve_output_output -o extracted_prompts

python extract_prompts.py examples/go_ramsey_search/openevolve_output -o examples/go_ramsey_search/openevolve_output/extracted_prompts
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def extract_prompts_from_json(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract prompts data from a program JSON object."""
    prompts = json_data.get("prompts", {})
    if not prompts:
        return {}
    
    extracted = {}
    for prompt_type, prompt_data in prompts.items():
        if isinstance(prompt_data, dict):
            extracted[prompt_type] = prompt_data
    
    return extracted


def save_prompt_section(content: str, output_path: Path, section_name: str):
    """Save a prompt section to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {section_name.upper()} ===\n\n")
        f.write(content)
        f.write("\n")


def process_program_json(json_file_path: Path, output_base: Path, 
                        checkpoint_name: str, program_id: str):
    """Process a single program JSON file and extract its prompts."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create output directory for this program
        program_output_dir = output_base / checkpoint_name / program_id
        
        # Extract basic program info
        info = {
            "id": data.get("id", "unknown"),
            "generation": data.get("generation", "unknown"),
            "timestamp": data.get("timestamp", "unknown"),
            "iteration_found": data.get("iteration_found", "unknown"),
            "metrics": data.get("metrics", {}),
            "parent_id": data.get("parent_id", "unknown")
        }
        
        # Save program info
        info_path = program_output_dir / "program_info.txt"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("=== PROGRAM INFORMATION ===\n\n")
            for key, value in info.items():
                if key == "metrics" and isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for metric_key, metric_value in value.items():
                        f.write(f"  {metric_key}: {metric_value}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
        
        # Extract and save prompts
        prompts = extract_prompts_from_json(data)
        
        if not prompts:
            print(f"No prompts found in {json_file_path}")
            return
        
        for prompt_type, prompt_data in prompts.items():
            prompt_dir = program_output_dir / prompt_type
            
            if isinstance(prompt_data, dict):
                # Handle structured prompt data
                for section, content in prompt_data.items():
                    if isinstance(content, str) and content.strip():
                        output_path = prompt_dir / f"{section}.txt"
                        save_prompt_section(content, output_path, f"{prompt_type}_{section}")
                    elif isinstance(content, list):
                        # Handle response arrays
                        for i, response in enumerate(content):
                            if isinstance(response, str) and response.strip():
                                output_path = prompt_dir / f"{section}_{i+1}.txt"
                                save_prompt_section(response, output_path, f"{prompt_type}_{section}_{i+1}")
                            elif isinstance(response, dict):
                                # Handle structured responses
                                for resp_key, resp_value in response.items():
                                    if isinstance(resp_value, str) and resp_value.strip():
                                        output_path = prompt_dir / f"{section}_{i+1}_{resp_key}.txt"
                                        save_prompt_section(resp_value, output_path, 
                                                          f"{prompt_type}_{section}_{i+1}_{resp_key}")
        
        print(f"Processed: {checkpoint_name}/{program_id}")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {json_file_path}: {e}")
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract prompts from OpenEvolve checkpoint files")
    parser.add_argument("input_dir", 
                      help="Path to openevolve_output_output directory")
    parser.add_argument("-o", "--output", default="extracted_prompts",
                      help="Output directory for extracted prompts (default: extracted_prompts)")
    parser.add_argument("--checkpoint", 
                      help="Process only specific checkpoint (e.g., checkpoint_5)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        return
    
    checkpoints_dir = input_path / "checkpoints"
    if not checkpoints_dir.exists():
        print(f"Error: Checkpoints directory {checkpoints_dir} does not exist")
        return
    
    # Process checkpoints
    processed_count = 0
    
    for checkpoint_dir in sorted(checkpoints_dir.iterdir()):
        if not checkpoint_dir.is_dir():
            continue
            
        checkpoint_name = checkpoint_dir.name
        
        # Skip if specific checkpoint requested and this isn't it
        if args.checkpoint and checkpoint_name != args.checkpoint:
            continue
            
        programs_dir = checkpoint_dir / "programs"
        if not programs_dir.exists():
            print(f"Warning: No programs directory in {checkpoint_name}")
            continue
        
        print(f"\nProcessing {checkpoint_name}...")
        
        # Process all JSON files in the programs directory
        json_files = list(programs_dir.glob("*.json"))
        if not json_files:
            print(f"Warning: No JSON files found in {programs_dir}")
            continue
            
        for json_file in json_files:
            program_id = json_file.stem  # filename without extension
            process_program_json(json_file, output_path, checkpoint_name, program_id)
            processed_count += 1
    
    print(f"\nProcessing complete! Processed {processed_count} program files.")
    print(f"Output saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main() 