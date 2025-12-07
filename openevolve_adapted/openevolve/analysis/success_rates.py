"""
Success rate analysis for OpenEvolve experiments

This module provides functions to compute extraction and validity success rates
from historical records, helping analyze the quality of LLM code generation.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def extract_iteration_from_dirname(dirname: str) -> Optional[int]:
    """Extract iteration number from directory name like 'iter_123_gen_04_abc123_175332'"""
    match = re.match(r'iter_(\d+)_', dirname)
    if match:
        return int(match.group(1))
    return None


def compute_extraction_success_rate(historical_dir: Path) -> Tuple[float, int, int]:
    """
    Compute the rate of successful extractions based on expected total iterations
    
    Args:
        historical_dir: Path to historical_records directory
        
    Returns:
        Tuple of (success_rate, successful_count, total_expected)
    """
    if not historical_dir.exists():
        logger.warning(f"Historical records directory not found: {historical_dir}")
        return 0.0, 0, 0
        
    iter_dirs = [d for d in historical_dir.iterdir() if d.is_dir() and d.name.startswith('iter_')]
    
    if not iter_dirs:
        logger.warning(f"No iteration directories found in {historical_dir}")
        return 0.0, 0, 0
    
    # Find the maximum iteration number to determine expected total
    max_iteration = -1
    iteration_extractions = {}  # iteration -> has_extraction
    
    for iter_dir in iter_dirs:
        iteration = extract_iteration_from_dirname(iter_dir.name)
        if iteration is None:
            continue
            
        max_iteration = max(max_iteration, iteration)
        
        # Check if this iteration has successful extraction
        extracted_prompts_dir = iter_dir / 'extracted_prompts'
        has_extraction = extracted_prompts_dir.exists() and any(extracted_prompts_dir.iterdir())
        
        # For each iteration, mark as successful if ANY of its directories have extraction
        if iteration not in iteration_extractions:
            iteration_extractions[iteration] = has_extraction
        else:
            iteration_extractions[iteration] = iteration_extractions[iteration] or has_extraction
    
    # Expected total iterations: 1 to max_iteration (excluding iter_0, using max checkpoint number)  
    expected_total_iterations = max_iteration if max_iteration >= 1 else 0
    # Only count extractions for iterations > 0
    successful_extractions = sum(has_extraction for iter_num, has_extraction in iteration_extractions.items() if iter_num > 0)
    
    success_rate = successful_extractions / expected_total_iterations if expected_total_iterations > 0 else 0.0
    logger.info(f"Extraction success rate: {success_rate:.2%} ({successful_extractions}/{expected_total_iterations})")
    
    return success_rate, successful_extractions, expected_total_iterations


def compute_validity_success_rate(historical_dir: Path) -> Tuple[float, int, int]:
    """
    Compute the rate of successful validity (program execution) based on expected total iterations
    
    Args:
        historical_dir: Path to historical_records directory
        
    Returns:
        Tuple of (success_rate, successful_count, total_expected)
    """
    if not historical_dir.exists():
        logger.warning(f"Historical records directory not found: {historical_dir}")
        return 0.0, 0, 0
        
    iter_dirs = [d for d in historical_dir.iterdir() if d.is_dir() and d.name.startswith('iter_')]
    
    if not iter_dirs:
        logger.warning(f"No iteration directories found in {historical_dir}")
        return 0.0, 0, 0
    
    # Find the maximum iteration number to determine expected total
    max_iteration = -1
    iteration_validities = {}  # iteration -> has_successful_validity
    
    for iter_dir in iter_dirs:
        iteration = extract_iteration_from_dirname(iter_dir.name)
        if iteration is None:
            continue
            
        max_iteration = max(max_iteration, iteration)
            
        metadata_file = iter_dir / 'metadata.json'
        if not metadata_file.exists():
            continue
            
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metrics = metadata.get('metrics', {})
            validity = metrics.get('validity', 0.0)
            
            # Consider it a successful validity if validity > 0
            has_successful_validity = validity and validity > 0
            
            # For each iteration, mark as successful if ANY of its directories have successful validity
            if iteration not in iteration_validities:
                iteration_validities[iteration] = has_successful_validity
            else:
                iteration_validities[iteration] = iteration_validities[iteration] or has_successful_validity
                
        except Exception as e:
            logger.debug(f"Failed to load {metadata_file}: {e}")
            continue
    
    # Expected total iterations: 1 to max_iteration (excluding iter_0, using max checkpoint number)
    expected_total_iterations = max_iteration if max_iteration >= 1 else 0
    # Only count validity for iterations > 0
    successful_validities = sum(has_validity for iter_num, has_validity in iteration_validities.items() if iter_num > 0)
    
    success_rate = successful_validities / expected_total_iterations if expected_total_iterations > 0 else 0.0
    logger.info(f"Validity success rate: {success_rate:.2%} ({successful_validities}/{expected_total_iterations})")
    
    return success_rate, successful_validities, expected_total_iterations


def compute_success_rates(output_dir: Path) -> Dict[str, Any]:
    """
    Compute all success rates for an experiment
    
    Args:
        output_dir: Path to experiment output directory
        
    Returns:
        Dictionary with success rate information
    """
    historical_dir = output_dir / "historical_records"
    
    if not historical_dir.exists():
        logger.warning(f"No historical_records found in {output_dir}")
        return {
            "extraction_rate": 0.0,
            "extraction_count": 0,
            "validity_rate": 0.0,
            "validity_count": 0,
            "total_expected_iterations": 0,
            "has_data": False
        }
    
    extraction_rate, extraction_count, total_iterations = compute_extraction_success_rate(historical_dir)
    validity_rate, validity_count, _ = compute_validity_success_rate(historical_dir)
    
    return {
        "extraction_rate": extraction_rate,
        "extraction_count": extraction_count,
        "validity_rate": validity_rate,
        "validity_count": validity_count,
        "total_expected_iterations": total_iterations,
        "has_data": total_iterations > 0
    }