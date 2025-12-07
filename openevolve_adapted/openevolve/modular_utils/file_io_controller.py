"""
Universal File I/O Controller for OpenEvolve Problems

Streamlined version containing only the essential functions actually used.
Provides universal save/load functions that work with any optimization problem.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from openevolve.modular_utils.yaml_config_service import create_problem_config_from_yaml


def save_search_results(best_perfect_solution: Optional[Dict],
                       current_solution: Dict, base_dir: str = ".",
                       directory_pattern: str = "{}_search_data", **kwargs) -> str:
    """
    Universal function to save search results for any problem type

    Args:
        best_perfect_solution: Best perfect solution found (None if none)
        current_solution: Current working solution
        base_dir: Base directory for saving. You'd better don't change it.
        directory_pattern: Pattern for directory naming, with {} placeholder for sanitized problem_type
        **kwargs: Problem-specific additional parameters

    Returns:
        Path to saved search_info.json file
    """
    try:
        # Get problem_type from config
        config = create_problem_config_from_yaml()
        problem_type = config.get('problem_type', 'Unknown')

        # Create problem-specific directory using configurable pattern
        sanitized_type = problem_type.lower().replace('(', '').replace(')', '').replace(',', '_').replace(' ', '_')
        dir_name = directory_pattern.format(sanitized_type)
        solutions_dir = os.path.join(base_dir, dir_name)
        os.makedirs(solutions_dir, exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        
        # Save solution files if they exist
        perfect_file = None
        current_file = None
        
        if best_perfect_solution and best_perfect_solution.get('data'):
            perfect_file = os.path.join(solutions_dir, f"solution_perfect_{timestamp}.json")
            with open(perfect_file, 'w') as f:
                json.dump(best_perfect_solution['data'], f)
        
        if current_solution.get('data'):
            current_file = os.path.join(solutions_dir, f"solution_current_{timestamp}.json")
            with open(current_file, 'w') as f:
                json.dump(current_solution['data'], f)
        
        # Create unified search_info structure
        search_info = {
            "timestamp": timestamp,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "problem_type": problem_type,
            "best_perfect_solution": {
                **best_perfect_solution,
                "data_file": perfect_file,
                "solution_id": f"perfect_{timestamp}"
            } if best_perfect_solution else None,
            "current_solution": {
                **current_solution,
                "data_file": current_file,
                "solution_id": f"current_{timestamp}"
            }
        }
        
        # Add any additional problem-specific fields
        search_info.update(kwargs)
        
        info_file = os.path.join(solutions_dir, "search_info.json")
        with open(info_file, 'w') as f:
            json.dump(search_info, f, indent=2)
        
        return info_file
        
    except Exception as e:
        print(f"Failed to save search results: {e}")
        return None


def extract_solution_data_universal(temp_dir: str = ".",
                                   directory_pattern: str = "{}_search_data") -> Dict[str, Any]:
    """
    Universal solution data extraction for any problem type

    Args:
        temp_dir: Directory to search for solution data
        directory_pattern: Pattern for directory naming, with {} placeholder for sanitized problem_type

    Returns:
        full_data dict containing current_solution and all metadata
    """
    try:
        # Get problem_type from config
        config = create_problem_config_from_yaml()
        problem_type = config.get('problem_type', 'Unknown')

        # Try new modular format first
        # Create directory name from problem type using configurable pattern
        sanitized_type = problem_type.lower().replace('(', '').replace(')', '').replace(',', '_').replace(' ', '_')
        dir_name = directory_pattern.format(sanitized_type)
        new_format_dir = os.path.join(temp_dir, dir_name)

        if os.path.exists(new_format_dir):
            search_info_file = os.path.join(new_format_dir, "search_info.json")
            if os.path.exists(search_info_file):
                with open(search_info_file, 'r') as f:
                    search_info = json.load(f)

                # Just return search_info directly
                return search_info

        raise FileNotFoundError("No modular search_info.json found")

        
    except Exception as e:
        # print(f"[Debug] Error extracting solution data: {e}")
        return {}


# Utility functions for common directory operations
def create_temp_directory(prefix: str = "openevolve_temp_") -> str:
    """Create a temporary directory for computation"""
    import tempfile
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_temp_directory(temp_dir: str):
    """Safely remove temporary directory"""
    import shutil
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Failed to cleanup temp directory {temp_dir}: {e}")


if __name__ == "__main__":
    # Simple test
    print("=== Testing Streamlined File I/O Controller ===")
    
    # Test save_search_results
    test_current = {
        'n': 100,
        'violations': 5,
        'data': [0, 1, 2, 3] * 25
    }
    
    test_perfect = {
        'n': 50,
        'violations': 0,
        'data': [0, 1, 2, 3] * 12 + [0, 1]
    }
    
    result_file = save_search_results("W(4,4)", test_perfect, test_current, "/tmp", test_param="example")
    print(f"Saved to: {result_file}")
    
    # Test extract_solution_data_universal
    if result_file:
        temp_dir = os.path.dirname(os.path.dirname(result_file))
        full_data = extract_solution_data_universal(temp_dir, "W(4,4)")
        solution_data = full_data.get('current_solution')
        print(f"Extracted: {len(solution_data) if solution_data else 0} elements, violations={full_data.get('violations', 'N/A')}")
    
    print("Streamlined File I/O test completed!")