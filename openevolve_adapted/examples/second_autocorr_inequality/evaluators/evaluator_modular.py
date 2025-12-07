import sys
import os
sys.path.append(os.path.dirname(__file__))
from openevolve.modular_utils.evaluation_controller import create_evaluation_controller, ObjectiveBasedProblemEvaluator, get_current_problem_config
from verify import validate_solution, compute_objective_value

# Get problem configuration from YAML
PROBLEM_CONFIG = get_current_problem_config()


def evaluate(program_path, temp_dir=None):
    """
    Modular problem evaluation using universal controller

    Note: Second autocorrelation is a MAXIMIZATION problem (maximize R(f))
    """
    problem_evaluator = ObjectiveBasedProblemEvaluator(
        validate_func=validate_solution,
        compute_objective_func=compute_objective_value,
        maximize=True  # MAXIMIZATION: higher R(f) is better
    )

    evaluation_controller = create_evaluation_controller(problem_evaluator, PROBLEM_CONFIG)

    return evaluation_controller.evaluate(program_path, temp_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = evaluate(sys.argv[1])
        print(f"Score: {result.metrics.get('combined_score')}")
        print(f"Achievement: {result.artifacts.get('achievement')}")
    else:
        print("Usage: python evaluator_modular.py <program_path>")