"""
Universal Scoring Controller for OpenEvolve Problems

Streamlined version supporting single-objective optimization.
Removed multi-objective and unused comparison functions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
# Import standardized error constants
from openevolve.modular_utils.error_constants import ErrorCodes, ErrorThresholds

EPS = 1e-8  # Small epsilon for numerical stability

@dataclass
class ScoreComponents:
    """Components of a solution score"""
    combined_score: float                          # Final combined score for ranking
    objective_value: Union[int, float, None] = None      # Primary objective (maximize/minimize)
    constraint_violations: int = 0                       # Number of constraint violations
    penalty_score: float = 0.0                          # Penalty for violations
    bonus_score: float = 0.0                             # Bonus for special achievements
    
    # Scoring metadata
    feasible: bool = True                                # Whether solution satisfies all constraints
    scoring_method: str = "default"                      # Method used for scoring
    
    def __post_init__(self):
        # Auto-set feasible based on violations
        if self.constraint_violations > 0:
            self.feasible = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScoreComponents':
        """Create from dictionary"""
        return cls(**data)


class ScoringFunction(ABC):
    """Abstract base class for scoring functions with unified error handling"""
    
    # Legacy compatibility - use new error constants
    EXECUTION_FAILURE_VIOLATIONS = abs(ErrorCodes.EXECUTION_FAILED)    # Program execution failed
    NO_SOLUTION_VIOLATIONS = abs(ErrorCodes.NO_SOLUTION_FOUND)         # Program ran but no solution found  
    VALIDATION_FAILURE_VIOLATIONS = abs(ErrorCodes.VALIDATION_FAILED)  # Solution format invalid
    
    def __init__(self):
        """Initialize scoring function"""
        pass
    
    @classmethod
    def is_execution_failure(cls, violations: int) -> bool:
        """Check if violation count indicates any kind of failure (not normal violations)"""
        if violations is None:
            return True  # None violations are always considered failures
        return ErrorThresholds.is_error_code(violations)
    
    @abstractmethod
    def get_failure_score(self, known_bound: Optional[int] = None) -> float:
        """Get appropriate failure score for this scoring method
        
        Args:
            known_bound: Problem-specific known bound
            
        Returns:
            Failure score appropriate for this scoring method
        """
        pass
    
    @abstractmethod
    def compute_score(self, solution_data: Any, 
                     violations: int = 0,
                     metadata: Optional[Dict[str, Any]] = None,
                     **kwargs) -> ScoreComponents:
        """
        Compute score for a solution
        
        Args:
            solution_data: The solution data (problem-specific format)
            violations: Number of constraint violations
            metadata: Additional metadata about the solution
            **kwargs: Problem-specific additional parameters
            
        Returns:
            ScoreComponents with detailed scoring breakdown
        """
        pass


class LinearObjectiveScoring(ScoringFunction):
    """
    Linear objective scoring with violation penalties
    Score = objective - violation_penalty * violations
    """
    
    def __init__(self, violation_penalty_factor: float = 1.0,
                 maximize_objective: bool = True):
        """
        Initialize linear scoring
        
        Args:
            violation_penalty_factor: Penalty multiplier for violations
            maximize_objective: Whether to maximize (True) or minimize (False) objective
        """
        super().__init__()
        self.violation_penalty_factor = violation_penalty_factor
        self.maximize_objective = maximize_objective
    
    def get_failure_score(self, known_bound: Optional[int] = None) -> float:
        """Get failure score for linear scoring - use standard large negative"""
        bound = known_bound or 1000
        return -bound  # Large negative for linear scoring
    
    def compute_score(self, solution_data: Any,
                     violations: int = 0,
                     metadata: Optional[Dict[str, Any]] = None,
                     **kwargs) -> ScoreComponents:
        """Compute linear score"""
        metadata = metadata or {}
        
        # Extract or compute objective value (keep raw value for metrics)
        if 'objective_value' in kwargs:
            objective_raw = kwargs['objective_value']  # Fixed: removed .deepcopy() - float is immutable
        elif hasattr(solution_data, '__len__'):
            # For sequences (like colorings), use length as default objective
            objective_raw = len(solution_data)
        elif 'length' in metadata:
            objective_raw = metadata['length']
        else:
            # Try to extract from kwargs or default to 0
            objective_raw = kwargs.get('length', 0)

        # Apply transformation for combined_score (1/x for minimize to keep success scores > 0)
        if not self.maximize_objective:
            if objective_raw <= 0:
                assert False, f"Objective value must be positive for minimization with reciprocal scoring, got {objective_raw}"
            objective_for_score = 1 / (objective_raw + EPS)
        else:
            objective_for_score = objective_raw

        # Calculate penalty
        penalty = self.violation_penalty_factor * violations

        # Combined score
        combined_score = objective_for_score - penalty

        return ScoreComponents(
            combined_score=combined_score,
            objective_value=objective_raw,  # Fixed: store raw value, not transformed
            constraint_violations=violations,
            penalty_score=penalty,
            feasible=(violations == 0),
            scoring_method="linear"
        )


class SOTARelativeScoring(ScoringFunction):
    """
    SOTA (State-of-the-Art) relative scoring
    Compares solution quality relative to known best results
    """
    
    def __init__(self, sota_baseline: float,
                 perfect_bonus_factor: float = 10.0,
                 improvement_bonus_factor: float = 5.0):
        """
        Initialize SOTA relative scoring
        
        Args:
            sota_baseline: Current state-of-the-art baseline value
            perfect_bonus_factor: Bonus multiplier for perfect solutions
            improvement_bonus_factor: Bonus multiplier for improvements over SOTA
        """
        super().__init__()
        self.sota_baseline = sota_baseline
        self.perfect_bonus_factor = perfect_bonus_factor
        self.improvement_bonus_factor = improvement_bonus_factor
    
    def get_failure_score(self, known_bound: Optional[int] = None) -> float:
        """Get failure score for SOTA relative scoring - moderately negative since scores are typically <=0"""
        bound = known_bound or 1000
        return -bound - 10  # Moderately negative for SOTA relative scoring
    
    def compute_score(self, solution_data: Any,
                     violations: int = 0,
                     metadata: Optional[Dict[str, Any]] = None,
                     **kwargs) -> ScoreComponents:
        """Compute SOTA relative score"""
        metadata = metadata or {}
        
        # Extract solution quality metric (problem-specific)
        if hasattr(solution_data, '__len__'):
            quality_metric = len(solution_data)
        else:
            quality_metric = kwargs.get('quality_metric', 0)
        
        # Base score calculation
        if violations == 0:
            # Perfect solution - high reward
            base_score = self.perfect_bonus_factor
            
            # Extra bonus if improving SOTA
            if quality_metric > self.sota_baseline:
                improvement = quality_metric - self.sota_baseline
                base_score += self.improvement_bonus_factor * improvement
            else:
                # Still good, scale based on quality relative to SOTA
                base_score += (quality_metric / self.sota_baseline)
        else:
            # Imperfect solution - scale down based on violations
            base_score = max(0.1, 1.0 - (violations / max(quality_metric, 1)))
        
        bonus = self.perfect_bonus_factor if violations == 0 else 0.0
        
        return ScoreComponents(
            combined_score=base_score,
            objective_value=quality_metric,
            constraint_violations=violations,
            bonus_score=bonus,
            feasible=(violations == 0),
            scoring_method="sota_relative"
        )


class DirectPerfectNScoring(ScoringFunction):
    """
    Direct perfect_n scoring: score ≈ perfect_n (intuitive and interpretable)
    Generic pattern for problems where primary objective is maximizing perfect solutions
    """
    
    def __init__(self, score_offset: float = 2.0,
                 violation_penalty_factor: float = 200.0,
                 max_penalty: float = 0.5):
        """
        Initialize direct perfect_n scoring
        
        Args:
            score_offset: Minimum score offset (score = perfect_n + offset)
            violation_penalty_factor: Penalty divisor for violations
            max_penalty: Maximum penalty for violations
        """
        super().__init__()
        self.score_offset = score_offset
        self.violation_penalty_factor = violation_penalty_factor
        self.max_penalty = max_penalty
    
    def get_failure_score(self, known_bound: Optional[int] = None) -> float:
        """Get failure score for direct perfect_n scoring - slightly negative since normal scores are >=1.5"""
        return -1.0  # Slightly negative for direct scoring (normal range is >=1.5)
    
    def compute_score(self, solution_data: Any,
                     violations: int = 0,
                     metadata: Optional[Dict[str, Any]] = None,
                     **kwargs) -> ScoreComponents:
        """
        Compute direct perfect_n score: score = (perfect_n + offset) - violation_penalty
        
        Expects search_info in metadata with best_perfect_solution containing n value.
        This gives intuitive scores where score ≈ perfect_n:
        - perfect_n=0, violations=0 → score = 2.0 (minimum)
        - perfect_n=42, violations=0 → score = 44.0  
        - perfect_n=1048, violations=0 → score = 1050.0
        - perfect_n=1049, violations=0 → score = 1051.0 (breakthrough!)
        """
        metadata = metadata or {}
        
        # Handle execution failure cases first
        if self.is_execution_failure(violations):
            # Program execution failed 
            return ScoreComponents(
                combined_score=self.get_failure_score(),
                objective_value=-1,  # Special perfect_n for execution failures
                constraint_violations=violations,
                penalty_score=0.0,
                feasible=False,
                scoring_method="direct_perfect_n_error"
            )
        
        # Get perfect_n from metadata (computed by problem-specific evaluator)
        perfect_n = metadata.get('perfect_n', 0)
        
        # Apply penalty for violations
        violation_penalty = min(violations / self.violation_penalty_factor, self.max_penalty)
        combined_score = (perfect_n + self.score_offset) - violation_penalty
        
        return ScoreComponents(
            combined_score=combined_score,
            objective_value=perfect_n,  # Primary objective is perfect_n
            constraint_violations=violations,
            penalty_score=violation_penalty,
            feasible=(violations == 0),
            scoring_method="direct_perfect_n"
        )


class VanillaObjectiveScoring(ScoringFunction):
    """
    Pure objective-based scoring for optimization problems
    Score = objective_value (no violation penalty)
    """
    
    def __init__(self, maximize: bool = True):
        """
        Initialize vanilla objective scoring
        
        Args:
            maximize: True to maximize objective, False to minimize
        """
        super().__init__()
        self.maximize = maximize
    
    def get_failure_score(self, known_bound: Optional[int] = None) -> float:
        """Get failure score for objective scoring"""
        if self.maximize:
            return -999999.0  # Large negative for maximization failures
        else:
            return 999999.0   # Large positive for minimization failures
    
    def compute_score(self, solution_data: Any,
                     violations: int = 0,
                     metadata: Optional[Dict[str, Any]] = None,
                     **kwargs) -> ScoreComponents:
        """Compute pure objective score"""
        metadata = metadata or {}
        
        # Handle execution failures
        if self.is_execution_failure(violations):
            failure_score = self.get_failure_score(metadata.get('known_bound'))
            return ScoreComponents(
                combined_score=failure_score,
                constraint_violations=violations,
                feasible=False,
                scoring_method=f"vanilla_objective_failure({'maximize' if self.maximize else 'minimize'})"
            )
        
        # Extract objective from metadata or solution_data
        objective_value = None
        if 'objective_value' in metadata:
            objective_value = metadata['objective_value']
        elif 'objective_value' in kwargs:
            objective_value = kwargs['objective_value']
        elif hasattr(solution_data, '__len__') and len(solution_data) == 3:
            # For 3-tuple solutions: (data1, data2, objective_value)
            objective_value = solution_data[2] if isinstance(solution_data, (tuple, list)) else None
        elif isinstance(solution_data, (int, float)):
            objective_value = solution_data
        
        if objective_value is None:
            # No objective found - return failure
            failure_score = self.get_failure_score(metadata.get('known_bound'))
            return ScoreComponents(
                combined_score=failure_score,
                constraint_violations=self.NO_SOLUTION_VIOLATIONS,
                feasible=False,
                scoring_method=f"vanilla_objective_no_objective({'maximize' if self.maximize else 'minimize'})"
            )
        
        # Pure objective score
        # final_score = objective_value if self.maximize else -objective_value
        final_score = objective_value if self.maximize else 1 / (objective_value + EPS)
        
        return ScoreComponents(
            combined_score=final_score,
            objective_value=objective_value,
            constraint_violations=violations,
            feasible=True,  # No constraint violations in pure objective scoring
            scoring_method=f"vanilla_objective({'maximize' if self.maximize else 'minimize'})"
        )


# Factory function
def create_scoring_controller(scoring_method: str = "linear", **kwargs) -> ScoringFunction:
    """Create scoring function with specified method"""
    if scoring_method == "linear":
        return LinearObjectiveScoring(**kwargs)
    elif scoring_method == "sota_relative":
        return SOTARelativeScoring(**kwargs)
    elif scoring_method == "direct_perfect_n":
        return DirectPerfectNScoring(**kwargs)
    elif scoring_method == "vanilla_objective":
        return VanillaObjectiveScoring(**kwargs)
    elif scoring_method == "objective":
        return VanillaObjectiveScoring(**kwargs)
    else:
        # Default to linear
        return LinearObjectiveScoring()


# Quick scoring convenience function
def score_solution(solution_data: Any, violations: int = 0, 
                  scoring_method: str = "linear", **kwargs) -> ScoreComponents:
    """Quick scoring with specified method"""
    scoring_func = create_scoring_controller(scoring_method, **kwargs)
    return scoring_func.compute_score(solution_data, violations, **kwargs)


if __name__ == "__main__":
    # Test the streamlined scoring controller
    print("=== Testing Streamlined Scoring Controller ===")
    
    # Test data
    test_solution = [0, 1, 2, 3] * 25  # 100-element solution
    
    # Test linear scoring
    linear_scorer = create_scoring_controller("linear", violation_penalty_factor=2.0)
    linear_score = linear_scorer.compute_score(test_solution, violations=5)
    print(f"✓ Linear scoring: score={linear_score.combined_score}, feasible={linear_score.feasible}")
    
    # Test SOTA relative scoring
    sota_scorer = create_scoring_controller("sota_relative", sota_baseline=50)
    sota_score = sota_scorer.compute_score(test_solution, violations=0)
    print(f"✓ SOTA scoring: score={sota_score.combined_score}, bonus={sota_score.bonus_score}")
    
    # Test convenience function
    quick_score = score_solution(test_solution, violations=2, scoring_method="linear")
    print(f"✓ Quick scoring: score={quick_score.combined_score}")
    
    print("Streamlined scoring controller test completed!")