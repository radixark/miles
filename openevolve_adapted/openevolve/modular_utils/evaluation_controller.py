"""
Universal Evaluation Controller

Provides standardized evaluation workflow with abstract interfaces for:
- Data extraction 
- Validation
- Scoring
- Error handling

Problem-specific implementations override abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, Callable
import time
# Import standardized error constants
from openevolve.evaluation_result import EvaluationResult
from openevolve.modular_utils.run_with_timeout_controller import create_timeout_controller
from openevolve.modular_utils.config_controller import BaseConfig
from openevolve.modular_utils.error_constants import ErrorCodes, ErrorMessages, DefaultValues
from openevolve.modular_utils.file_io_controller import extract_solution_data_universal
from openevolve.modular_utils.scoring_controller import DirectPerfectNScoring, VanillaObjectiveScoring
from openevolve.modular_utils.yaml_config_service import create_problem_config_from_yaml
from openevolve.modular_utils.score_transform import ScoreTransformConfig, transform_score_for_rl, create_score_transform_config_from_dict
from openevolve.utils.metrics_utils import create_evaluation_metrics

class BaseProblemEvaluator(ABC):
    """Simplified abstract interface for problem evaluation"""
    
    @abstractmethod
    def validate_solution(self, solution_data: Any) -> Tuple[bool, str]:
        """Validate solution format and constraints
        Returns: (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def compute_objective_score(self, full_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Compute final score from full_data (includes solution, metadata, search_info)
        Handles all validation and scoring logic internally
        Returns: (combined_score, kwargs_dict)
        """
        pass


class BoundBasedProblemEvaluator(BaseProblemEvaluator):
    """Standard evaluator for bound-based optimization problems
    Handles perfect_n from best_perfect + violation penalties from current
    """

    def __init__(self, validate_func: Callable, count_violations_func: Callable,
                 cal_bound_func: Callable, violation_penalty_factor: float = 100.0,
                 score_offset: float = 2.0):
        assert False, "[TODO]: Implement BoundBasedProblemEvaluator"
        self.validate_solution_func = validate_func
        self.count_violations_func = count_violations_func
        self.cal_bound_func = cal_bound_func
        self.violation_penalty_factor = violation_penalty_factor
        self.score_offset = score_offset

    def validate_solution(self, solution_data: Any) -> Tuple[bool, str]:
        return self.validate_solution_func(solution_data)
    
    def compute_objective_score(self, full_data: Dict[str, Any]) -> float:
        """BoundBased scoring: perfect_n - violation_penalty using DirectPerfectNScoring"""
        
        # Extract solution data from search_info structure
        current_solution = full_data.get('current_solution', {})
        solution_data = current_solution.get('data')

        if not solution_data:
            raise ValueError("No current solution data found")

        # Step 1: Validate current solution
        is_valid, error_msg = self.validate_solution_func(solution_data)
        if not is_valid:
            raise ValueError(f"Validation failed: {error_msg}")

        # Step 2: Compute verified perfect_n from best_perfect
        verified_perfect_n = 0
        best_perfect_solution = full_data.get('best_perfect_solution')
        if best_perfect_solution and best_perfect_solution.get('data'):
            best_perfect_data = best_perfect_solution['data']
            is_valid, _ = self.validate_solution_func(best_perfect_data)
            if is_valid:
                violations = self.count_violations_func(best_perfect_data)
                if violations == 0:
                    verified_perfect_n = self.cal_bound_func(best_perfect_data)

        # Step 3: Compute current violations
        current_violations = self.count_violations_func(solution_data)
        
        # Step 4: Use DirectPerfectNScoring
        scorer = DirectPerfectNScoring(violation_penalty_factor=self.violation_penalty_factor, score_offset=self.score_offset)
        metadata = {'perfect_n': verified_perfect_n}
        score_components = scorer.compute_score(solution_data, current_violations, metadata)
        
        return score_components.combined_score, {'violations': current_violations, 'perfect_n': verified_perfect_n}


class ObjectiveBasedProblemEvaluator(BaseProblemEvaluator):
    """Standard evaluator for objective-based optimization problems
    Pure objective optimization without penalties
    """

    def __init__(self, validate_func: Callable, compute_objective_func: Callable,
                 score_transform_config: ScoreTransformConfig = None,
                 maximize: bool = True):
        """
        Args:
            validate_func: Function to validate solution format
            compute_objective_func: Function to compute objective value
            score_transform_config: Optional score transformation for RL training
            maximize: True to maximize objective (higher is better), False to minimize (lower is better)
        """
        self.validate_solution_func = validate_func
        self.compute_objective_func = compute_objective_func
        self.score_transform_config = score_transform_config
        self.maximize = maximize

    def validate_solution(self, solution_data: Any) -> Tuple[bool, str]:
        return self.validate_solution_func(solution_data)
    
    def compute_objective_score(self, full_data: Dict[str, Any]) -> float:
        """ObjectiveBased scoring: pure objective value using VanillaObjectiveScoring

        For minimization problems (maximize=False):
        - compute_objective_func returns the raw objective (e.g., 1.4581)
        - VanillaObjectiveScoring negates it to combined_score (e.g., -1.4581)
        - Database stores -1.4581 (higher/less negative is better)
        - objective_value remains 1.4581 (the actual value to minimize)
        """

        # Extract solution data from search_info structure
        current_solution = full_data.get('current_solution', {})
        solution_data = current_solution.get('data')

        if not solution_data:
            raise ValueError("No current solution data found")

        # Step 1: Validate solution
        is_valid, error_msg = self.validate_solution_func(solution_data)
        if not is_valid:
            raise ValueError(f"Validation failed: {error_msg}")

        # Step 2: Compute objective (with anti-cheat)
        objective_value = self.compute_objective_func(solution_data)

        # Step 3: Use VanillaObjectiveScoring with configured maximize/minimize mode
        scorer = VanillaObjectiveScoring(maximize=self.maximize)
        metadata = {'objective_value': objective_value}
        score_components = scorer.compute_score(solution_data, 0, metadata)  # 0 violations for valid solutions

        # combined_score is used by database (may be negated for minimize mode)
        combined_score = score_components.combined_score
        # objective_value is the raw value (always in original direction)
        kwargs = {'objective_value': objective_value}

        # Add RL-specific normalized reward if score transformation is configured
        if self.score_transform_config is not None:
            # IMPORTANT: transform expects RAW objective_value, not negated combined_score
            rl_normalized_reward = transform_score_for_rl(objective_value, self.score_transform_config)
            kwargs['rl_normalized_reward'] = rl_normalized_reward

        return combined_score, kwargs
    


class UniversalEvaluationController:
    """Universal evaluation controller with standardized error handling"""
    
    def __init__(self, base_problem_evaluator: BaseProblemEvaluator, config: Union[Dict[str, Any], BaseConfig]):
        self.base_problem_evaluator = base_problem_evaluator

        # Convert BaseConfig to dict if needed
        if isinstance(config, BaseConfig):
            self.config = {
                'time_limit': config['time_limit'],
                'known_bound': config.get('known_bound', 0),
                **config.to_dict()
            }
        else:
            self.config = config

        # Configure score transformation if it's an ObjectiveBasedProblemEvaluator
        if isinstance(base_problem_evaluator, ObjectiveBasedProblemEvaluator):
            base_problem_evaluator.score_transform_config = create_score_transform_config_from_dict(self.config)

        self.timeout_controller = create_timeout_controller(timeout=self.config.get('time_limit', 300))
    
    def create_error_result(self, error_type: str, error_message: str,
                          runtime_seconds: float = 0.0, timeout_occurred: bool = False,
                          program_output: str = "", program_stderr: str = "", **kwargs) -> EvaluationResult:
        """Create standardized error result"""
        # Map error types to standardized error codes
        error_code_mapping = {
            "execution_failed": ErrorCodes.EXECUTION_ERROR,
            "no_solution": ErrorCodes.EXECUTION_ERROR,
            "validation_failed": ErrorCodes.VALIDATION_FAILED,
            "timeout": ErrorCodes.EXECUTION_ERROR,
        }

        # Get appropriate error code or use generic execution failure
        error_code = error_code_mapping.get(error_type, ErrorCodes.EXECUTION_ERROR)
        error_score = float(error_code)  # Use error code directly as score

        # Get score_transform_config to ensure rl_normalized_reward is generated even for errors
        score_transform_config = None
        if isinstance(self.base_problem_evaluator, ObjectiveBasedProblemEvaluator):
            score_transform_config = self.base_problem_evaluator.score_transform_config

        # Use unified metrics creation function
        metrics = create_evaluation_metrics(
            combined_score=error_score,
            validity=0.0,  # Error case
            runtime_seconds=runtime_seconds,
            eval_time=runtime_seconds,
            exit_code=-1 if timeout_occurred else 1,
            timeout_occurred=timeout_occurred,
            score_transform_config=score_transform_config,
            **kwargs
        )

        return EvaluationResult(
            metrics=metrics,
            artifacts={
                "error": error_message,
                "error_type": error_type,
                # "program_output": program_output,
                # "program_stderr": program_stderr,
            }
        )
    
    def create_success_result(self, combined_score: float,
                            execution_result, eval_time: float, **kwargs) -> EvaluationResult:
        """Create standardized success result using full_data and kwargs"""

        # No need for score_transform_config as kwargs already contains rl_normalized_reward
        metrics = create_evaluation_metrics(
            combined_score=combined_score,
            validity=1.0,  # Success case
            runtime_seconds=execution_result.runtime_seconds,
            eval_time=eval_time,
            exit_code=execution_result.exit_code,
            timeout_occurred=execution_result.timeout_occurred,
            score_transform_config=None,  # kwargs already has rl_normalized_reward
            **kwargs
        )

        return EvaluationResult(
            metrics=metrics,
            artifacts={"program_output": execution_result.stdout}
        )
    
    def evaluate(self, program_path: str, temp_dir: Optional[str] = None) -> EvaluationResult:
        """Universal evaluation workflow"""
        start_time = time.time()
        
        try:
            # Run program with timeout
            if temp_dir:
                # just support python now
                execution_result = self.timeout_controller.run_python_program(
                    program_path=program_path,
                    working_directory=temp_dir,
                    timeout_seconds=self.config['time_limit']
                )
            else:
                execution_result, temp_dir = self.timeout_controller.run_in_temp_workspace(
                    program_path=program_path,
                    timeout_seconds=self.config['time_limit']
                )
            
            # Check execution success
            if not execution_result.success:
                if execution_result.timeout_occurred:
                    error_msg = f"Program timed out after {self.config['time_limit']}s"
                else:
                    error_parts = [f"Program execution failed (exit code: {execution_result.exit_code})"]
                    if execution_result.stderr.strip():
                        error_parts.append(f"Error: {execution_result.stderr.strip()}")
                    if execution_result.stdout.strip():
                        error_parts.append(f"Output: {execution_result.stdout.strip()}")
                    error_msg = " | ".join(error_parts)
                
                return self.create_error_result(
                    "execution_failed", error_msg,
                    execution_result.runtime_seconds, 
                    execution_result.timeout_occurred,
                    execution_result.stdout,
                    execution_result.stderr
                )
            
            # Extract full data using universal extractor  
            full_data = extract_solution_data_universal(temp_dir)
            
            if not full_data or not (full_data.get('current_solution') and full_data['current_solution'].get('data')):
                return self.create_error_result(
                    "no_solution", "No solution found in program output",
                    execution_result.runtime_seconds, False, execution_result.stdout
                )
            
            # Compute objective score (handles all validation and scoring internally)
            try:
                combined_score, score_kwargs = self.base_problem_evaluator.compute_objective_score(full_data)
            except ValueError as e:
                return self.create_error_result(
                    "validation_failed", str(e),
                    execution_result.runtime_seconds, False, execution_result.stdout
                )
            except Exception as e:
                return self.create_error_result(
                    "execution_failed", f"Scoring failed: {str(e)}",
                    execution_result.runtime_seconds, False, execution_result.stdout
                )
            
            # Extract basic info for reporting
            solution_data = full_data.get('current_solution', {}).get('data')

            eval_time = time.time() - start_time
            return self.create_success_result(
                combined_score, 
                execution_result, eval_time, **score_kwargs
            )
            
        except Exception as e:
            import traceback
            error_msg = f"Universal evaluation failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            
            return EvaluationResult(
                metrics={
                    "combined_score": DefaultValues.get_error_score(self.config.get('known_bound')),
                    "violations": int(ErrorCodes.EXECUTION_FAILED),  # Use standardized error code
                    "fitness": 0.0,
                    "eval_time": time.time() - start_time,
                    "validity": 0.0
                },
                artifacts={
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                }
            )


def create_evaluation_controller(base_problem_evaluator: BaseProblemEvaluator, 
                               config: Union[Dict[str, Any], BaseConfig]) -> UniversalEvaluationController:
    """Factory function to create evaluation controller"""
    return UniversalEvaluationController(base_problem_evaluator, config)


def get_current_problem_config() -> BaseConfig:
    """Get current problem configuration from YAML
    
    This is the main entry point for all problems to get their configuration.
    Replaces the need for individual config.py files in each problem directory.
    
    Returns:
        BaseConfig instance with configuration from current YAML file
    """
    return create_problem_config_from_yaml()