"""
Universal Configuration Controller for OpenEvolve Problems

Provides base configuration structure that all optimization problems can inherit from.
Contains ONLY universal functionality, no problem-specific implementations.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union


class BaseConfig:
    """Base configuration structure for any optimization problem"""
    
    def __init__(self, 
                 problem_type: str,
                 core_parameters: Dict[str, Any],
                 time_limit: int = 300,
                 known_bounds: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize base configuration
        
        Args:
            problem_type: Problem identifier (e.g., 'W(4,4)', 'H(100)', 'K(11)')
            core_parameters: Problem-specific core parameters
            time_limit: Execution time limit in seconds
            known_bounds: Known bounds/targets for the problem type
            **kwargs: Additional problem-specific configuration
        """
        self.config = {
            'problem_type': problem_type,
            'core_parameters': core_parameters,
            'time_limit': time_limit,
            'known_bounds': known_bounds or {},
            **kwargs
        }
        
        # Auto-derive known_bound for current problem
        self.config['known_bound'] = self.config['known_bounds'].get(problem_type, None)
        
        # Let subclass customize target values
        self._auto_derive_targets()
    
    def _auto_derive_targets(self):
        """Override in subclass to auto-derive target values based on known_bound"""
        return self.config["known_bound"] if self.config["known_bound"] is not None else None
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access: config['key']"""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Allow dict-style setting: config['key'] = value"""
        self.config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get with default value"""
        return self.config.get(key, default)
    
    def update(self, other_dict: Dict[str, Any]):
        """Update config with another dict"""
        self.config.update(other_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return as plain dict"""
        return self.config.copy()
    
