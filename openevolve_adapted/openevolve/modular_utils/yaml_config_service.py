"""
YAML Configuration Service for OpenEvolve Problems

Provides centralized configuration loading from YAML files for all optimization problems.
Eliminates need for individual problem config.py files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from openevolve.modular_utils.config_controller import BaseConfig


def get_current_yaml_path() -> Optional[Path]:
    """Get current YAML configuration file path from environment
    
    OpenEvolve sets OPENEVOLVE_CONFIG_PATH when running programs
    
    Returns:
        Path to current YAML config file or None if not found
    """
    # Check environment variable set by OpenEvolve
    config_path = os.environ.get('OPENEVOLVE_CONFIG_PATH')
    if config_path and Path(config_path).exists():
        return Path(config_path)
    
    # Fallback: look for yaml files in current working directory
    cwd = Path.cwd()
    yaml_files = list(cwd.glob("*.yaml")) + list(cwd.glob("*.yml"))
    if yaml_files:
        return yaml_files[0]
    
    return None


def load_yaml_variables(yaml_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load variables section from YAML configuration file
    
    Args:
        yaml_path: Path to YAML file. If None, auto-detect current config
        
    Returns:
        Dictionary of variables from YAML file, empty dict if not found
    """
    if yaml_path is None:
        yaml_path = get_current_yaml_path()
    
    if not yaml_path or not Path(yaml_path).exists():
        return {}
    
    try:
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            return yaml_config.get('variables', {})
    except Exception as e:
        print(f"Warning: Failed to load YAML config from {yaml_path}: {e}")
        return {}


def create_problem_config_from_yaml(yaml_path: Optional[Union[str, Path]] = None) -> BaseConfig:
    """Create BaseConfig from YAML variables section
    
    Args:
        yaml_path: Path to YAML file. If None, auto-detect current config
        
    Returns:
        BaseConfig instance with configuration from YAML or default values
    """
    variables = load_yaml_variables(yaml_path)
    
    if not variables:
        # Fallback to default configuration
        print("Warning: No YAML variables found, using default config")
        return BaseConfig(
            problem_type="Unknown",
            core_parameters={}
        )
    
    # Extract configuration from YAML variables
    # Require standard core_parameters structure
    core_parameters = variables.get('core_parameters', {})
    
    if not core_parameters:
        # No backward compatibility - YAML must use standard structure
        print("Warning: No 'core_parameters' found in YAML. Please use standard structure: variables.core_parameters")
        core_parameters = {}
    
    # Extract other configuration
    problem_type = variables.get('PROBLEM_TYPE', 'Unknown')
    known_bounds = variables.get('KNOWN_BOUNDS', {})
    max_runtime = variables.get('MAX_RUNTIME', 300)
    
    # Additional parameters
    kwargs = {}
    current_sota = variables.get('CURRENT_SOTA')
    if current_sota is not None:
        kwargs['current_sota'] = current_sota
    
    # Get target_value from core_parameters or top-level
    target_value = core_parameters.get('target_value') or variables.get('TARGET_VALUE')
    if target_value is not None:
        kwargs['target_value'] = target_value

    # Include score_transform configuration
    score_transform = variables.get('score_transform')
    if score_transform is not None:
        kwargs['score_transform'] = score_transform

    config = BaseConfig(
        problem_type=problem_type,
        core_parameters=core_parameters,
        time_limit=max_runtime,
        known_bounds=known_bounds,
        **kwargs
    )
    
    # Auto-derive additional parameters based on problem type
    known_bound = config.get('known_bound')
    if known_bound is not None:
        config['target_n'] = known_bound + 1
    
    return config


def get_problem_parameter(param_name: str, default: Any = None) -> Any:
    """Get specific parameter from current YAML configuration
    
    Args:
        param_name: Parameter name to retrieve
        default: Default value if not found
        
    Returns:
        Parameter value or default
    """
    variables = load_yaml_variables()
    
    # Try core_parameters first
    core_params = variables.get('core_parameters', {})
    if param_name in core_params:
        return core_params[param_name]
    
    # Try direct variables
    return variables.get(param_name, default)


if __name__ == "__main__":
    # Test the YAML configuration service
    print("=== Testing YAML Configuration Service ===")
    
    # Test loading configuration
    config = create_problem_config_from_yaml()
    print(f"✓ Created config: {config['problem_type']}")
    print(f"  Core parameters: {config['core_parameters']}")
    print(f"  Time limit: {config['time_limit']}")
    print(f"  Known bounds: {config.get('known_bounds', {})}")
    
    # Test parameter retrieval
    r_value = get_problem_parameter('r', 'not_found')
    print(f"✓ Parameter 'r': {r_value}")
    
    print("YAML configuration service test completed!")