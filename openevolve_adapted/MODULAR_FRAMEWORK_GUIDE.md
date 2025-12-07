# Adding New Problems to OpenEvolve Modular Framework

This guide provides a complete walkthrough for adding new optimization problems to OpenEvolve's modular framework, with templates and examples.

## Problem Types

The framework supports two types of problems:

### 1. Bound-Based Problems (Example: Van der Waerden Numbers)
- **Goal**: Find maximum N with zero constraint violations
- **Scoring**: Based on perfect_n achieved relative to known bounds
- **Evaluator**: `BoundBasedProblemEvaluator`
- **Required functions**: `validate_solution()`, `count_violations()`, `cal_bound()`

### 2. Objective-Based Problems (Example: Circle Packing)
- **Goal**: Maximize an objective value (e.g., sum of radii)
- **Scoring**: Direct objective value optimization
- **Evaluator**: `ObjectiveBasedProblemEvaluator`  
- **Required functions**: `validate_solution()`, `compute_objective_value()`

## Required Directory Structure

```
examples/YOUR_PROBLEM_NAME/
├── configs/
│   └── config_your_problem.yaml           # Problem configuration
├── evaluators/
│   ├── evaluator_modular.py               # Standard template (minimal changes)
│   └── verify.py                          # Problem-specific validation & scoring
├── initial_programs/
│   ├── initial_program_modular.py         # Problem-specific algorithm
│   └── ref/                               # (Optional) Reference data for programs to import
│       └── data.py                        # Reference data or utilities
└── run.sh                                 # Execution script (minimal changes)
```

### Reference Data (ref/) Directory
If your problem requires reference data (e.g., SOTA solutions, utilities):
- Place files in `initial_programs/ref/`
- Programs can import via: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ref")); from data import ...`
- Configure `copy_folders: ["initial_programs/ref"]` in YAML to copy during evaluation

## Implementation Steps

### Step 1: Create `verify.py` (Problem-Specific)

Choose your problem type and implement the required functions:

#### For Bound-Based Problems:

```python
"""
YOUR_PROBLEM Verification Functions
"""

from typing import List, Tuple, Any
from openevolve.modular_utils.evaluation_controller import get_current_problem_config

# Get problem configuration from YAML
PROBLEM_CONFIG = get_current_problem_config()


def validate_solution(solution_data: Any) -> Tuple[bool, str]:
    """
    Validate solution format and basic constraints

    Args:
        solution_data: Your problem's solution format (e.g., List[int] for coloring)

    Returns:
        (is_valid, error_message)
    """
    # TODO: Implement validation logic specific to your problem
    # Example checks:
    # - Correct data type and format
    # - Valid value ranges 
    # - Correct dimensions/length
    # - Basic constraint satisfaction
    
    # Example from VDW:
    if not isinstance(solution_data, list):
        return False, "Solution must be a list"
    
    # Add your specific validation logic here
    return True, ""


def count_violations(solution_data: Any) -> int:
    """
    Count constraint violations in the solution

    THIS FUNCTION MUST RECOMPUTE VIOLATIONS FROM solution_data
    DO NOT trust any pre-computed violation counts to prevent hacking

    Args:
        solution_data: Your problem's solution format

    Returns:
        Number of constraint violations (0 = perfect solution)
    """
    # TODO: Implement violation counting logic
    # SECURITY: Always recompute from raw solution_data
    
    # Example from VDW:
    # violations = 0
    # for each constraint:
    #     if constraint_violated(solution_data):
    #         violations += 1
    # return violations
    
    return 0  # Replace with actual implementation


def cal_bound(solution_data: Any) -> int:
    """
    Calculate the bound value for the problem

    Args:
        solution_data: Your problem's solution format

    Returns:
        Bound value - for minimization problems, return negative values
    """
    # TODO: Implement bound calculation logic
    # Examples:
    # - For VDW (maximization): return len(solution_data)  # length of coloring
    # - For Golomb ruler (minimization): return -max(solution_data)  # negative ruler length

    return len(solution_data)  # Replace with actual implementation
```

#### For Objective-Based Problems:

```python
"""
YOUR_PROBLEM Verification Functions
"""

from typing import List, Tuple, Any
import numpy as np


def validate_solution(solution_data: Any) -> Tuple[bool, str]:
    """
    Validate solution format and basic constraints

    Args:
        solution_data: Your problem's solution format

    Returns:
        (is_valid, error_message)
    """
    # TODO: Implement validation logic
    # Example from Circle Packing:
    if not isinstance(solution_data, (tuple, list)) or len(solution_data) != 2:
        return False, "Solution must be (data1, data2) tuple"
    
    # Add your specific validation logic here
    return True, ""


def compute_objective_value(solution_data: Any) -> float:
    """
    Compute objective value from solution

    THIS FUNCTION MUST RECOMPUTE THE OBJECTIVE FROM solution_data
    DO NOT trust any pre-computed objective values to prevent hacking

    Args:
        solution_data: Your problem's solution format

    Returns:
        Objective value to maximize
    """
    # TODO: Implement objective computation
    # SECURITY: Always recompute from raw solution_data
    
    # Example from Circle Packing:
    # centers, radii = solution_data
    # return float(np.sum(radii))  # Sum of radii
    
    return 0.0  # Replace with actual implementation
```

### Step 2: Create `evaluator_modular.py` (Standard Template)

#### For Bound-Based Problems:

```python
import sys
from openevolve.modular_utils.evaluation_controller import create_evaluation_controller, BoundBasedProblemEvaluator, get_current_problem_config
from examples.YOUR_PROBLEM_NAME.evaluators.verify import validate_solution, count_violations, cal_bound

# Get problem configuration from YAML
PROBLEM_CONFIG = get_current_problem_config()


def evaluate(program_path, temp_dir=None):
    """
    Modular problem evaluation using universal controller
    """
    problem_evaluator = BoundBasedProblemEvaluator(
        validate_func=validate_solution,
        count_violations_func=count_violations,
        cal_bound_func=cal_bound,
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
```

#### For Objective-Based Problems:

```python
import sys
from openevolve.modular_utils.evaluation_controller import create_evaluation_controller, ObjectiveBasedProblemEvaluator, get_current_problem_config
from examples.YOUR_PROBLEM_NAME.evaluators.verify import validate_solution, compute_objective_value

# Get problem configuration from YAML
PROBLEM_CONFIG = get_current_problem_config()


def evaluate(program_path, temp_dir=None):
    """
    Modular problem evaluation using universal controller
    """
    problem_evaluator = ObjectiveBasedProblemEvaluator(
        validate_func=validate_solution,
        compute_objective_func=compute_objective_value,
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
```

### Step 3: Create Configuration YAML File

#### Required Variables Structure:

```yaml
# OpenEvolve Configuration for YOUR_PROBLEM
variables:
  # Core problem parameters (REQUIRED)
  core_parameters:
    # TODO: Add your problem-specific parameters
    param1: value1
    param2: value2
    target_value: 1000  # For objective-based problems
  
  # Problem identifier (REQUIRED)
  PROBLEM_TYPE: "YourProblem_Description"
  
  # Optional: Known bounds database
  KNOWN_BOUNDS:
    "YourProblem_Instance1": 1000
    "YourProblem_Instance2": 2000
  
  # Optional: Runtime constraints  
  MAX_RUNTIME: 300

# Standard OpenEvolve configuration
max_iterations: 100
checkpoint_interval: 10
max_code_length: 50000
log_level: "INFO"

# LLM configuration
llm:
  models:
    - name: "google/gemini-2.5-flash-lite"
      weight: 1.0
  api_base: "https://openrouter.ai/api/v1"
  api_key: "your-api-key"
  temperature: 0.7
  top_p: 0.95
  max_tokens: 65536
  timeout: 600
  retries: 3
  retry_delay: 5

# Prompt configuration
prompt:
  use_alphaevolve_style: true
  use_system_prompt: false
  use_system_message_sampling: true
  system_message_list:
    - message: |
        You are an expert in YOUR_PROBLEM_DOMAIN. Your task is to improve algorithms for {PROBLEM_TYPE}.

        Problem parameters:
        - Parameter 1: {core_parameters.param1}
        - Parameter 2: {core_parameters.param2}
        - Target: {target_value}

        # TODO: Add problem-specific guidance

      weight: 1.0
    # Example: Golomb ruler with dual strategy messages
    # - message: |
    #     You are an expert in Golomb rulers. Focus on CONSTRUCTION approaches.
    #
    #     A Golomb ruler must start at 0, be strictly increasing, and have all pairwise distances unique.
    #     For {core_parameters.num_marks} marks, minimize ruler length.
    #
    #     Use systematic construction methods like modular arithmetic patterns.
    #   weight: 0.6
    # - message: |
    #     You are an expert in Golomb rulers. Focus on SEARCH approaches.
    #
    #     A Golomb ruler must start at 0, be strictly increasing, and have all pairwise distances unique.
    #     For {core_parameters.num_marks} marks, minimize ruler length.
    #
    #     Use optimization techniques like backtracking, local search, or metaheuristics.
    #   weight: 0.4

  num_top_programs: 1
  num_diverse_programs: 1
  num_inspiration_programs: 0
  use_template_stochasticity: true
  include_artifacts: true
  max_artifact_bytes: 16384
  artifact_security_filter: true

# Database configuration (MAP-Elites algorithm)
database:
  population_size: 70
  archive_size: 30
  num_islands: 5
  feature_dimensions:
    - "score"
    - "complexity"
  elite_selection_ratio: 0.3
  exploitation_ratio: 0.6
  log_prompts: true

# Evaluation settings
evaluator:
  timeout: 200
  max_retries: 3
  cascade_evaluation: false
  parallel_evaluations: 4
  use_llm_feedback: false
  enable_artifacts: true
  collect_runtime_environments: true
  preserve_temp_directories: false
  runtime_environment_patterns:
    - "*"
  copy_folders:
    - "initial_programs/ref"  # Optional: if you have reference data for programs to import

# Evolution settings
diff_based_evolution: true
allow_full_rewrites: false
```

### Step 4: Create `initial_program_modular.py` (Problem-Specific)

Remember you have to add `save_search_results` and use `get_current_problem_config` to load parameters from config. For the configs, you'd better put them in `__main__`, we find it would give better evolution performance than putting it at the top of the file.

```python

# EVOLVE-BLOCK-START
def your_main_algorithm():
    """
    Main algorithm for YOUR_PROBLEM
    
    This function will be evolved by OpenEvolve.
    Implement your problem-specific optimization logic here.
    
    Returns:
        solution_data: Your problem's solution format
    """
    # TODO: Implement your algorithm
    
    # Example structure:
    # 1. Initialize solution
    # 2. Optimize/search
    # 3. Return best solution found
    
    solution_data = None  # Replace with actual solution
    return solution_data

# EVOLVE-BLOCK-END

def util_func():
    """
    Some helper function
    """
    pass


def initial_program():
    """
    Main entry point called by OpenEvolve for YOUR_PROBLEM.
    This function must remain unchanged.
    """
    solution_data = your_main_algorithm()
    
    # Save search results using modular file_io
    current_solution = {'data': solution_data}
    best_perfect = None  # For bound-based: set if perfect solution found
    
    # have to include this saving function
    save_search_results(
        best_perfect,
        current_solution,
        # TODO: Add problem-specific metadata
        **{'param1': PARAM1, 'param2': PARAM2}
    )
    
    return solution_data


if __name__ == "__main__":
    # Import modular funcs
    from openevolve.modular_utils.file_io_controller import save_search_results
    from openevolve.modular_utils.evaluation_controller import get_current_problem_config

    ######## get parameters from config ########
    PROBLEM_CONFIG = get_current_problem_config()
    # TODO: Extract your problem-specific parameters
    PARAM1 = PROBLEM_CONFIG['core_parameters']['param1']
    PARAM2 = PROBLEM_CONFIG['core_parameters']['param2']
    PROBLEM_TYPE = PROBLEM_CONFIG['problem_type']
    TARGET_VALUE = PROBLEM_CONFIG.get('target_value')  # For objective-based problems
    ########################################

    result = initial_program()
    print(f"Generated {PROBLEM_TYPE} solution")
    # TODO: Add problem-specific output
```

### Step 5: Create `run.sh` (Standard Template)

```bash
#!/bin/bash

# YOUR_PROBLEM evolution run script (modular framework)
EXAMPLE=YOUR_PROBLEM_NAME
INIT_NAME="initial_program_modular.py"
EVAL_NAME="evaluator_modular.py"
CONFIG_NAME="config_your_problem.yaml"
ITER=50
POSTFIX="modular_test"

echo "Running YOUR_PROBLEM evolution with streamlined modular controllers..."
echo "Target: $ITER steps to test the system"

python openevolve-run.py examples/$EXAMPLE/initial_programs/$INIT_NAME \
  examples/$EXAMPLE/evaluators/$EVAL_NAME \
  --config examples/$EXAMPLE/configs/$CONFIG_NAME \
  --output examples/$EXAMPLE/outputs/openevolve_output_$POSTFIX \
  --iterations $ITER

echo "YOUR_PROBLEM evolution completed!"
```

## Configuration Requirements

### Required YAML Variables (Must Include):

1. **`variables.core_parameters`**: Dict containing problem-specific parameters
2. **`variables.PROBLEM_TYPE`**: String identifier for your problem
3. **Standard OpenEvolve sections**: `llm`, `prompt`, `database`, `evaluator`

### Optional YAML Variables:

1. **`variables.KNOWN_BOUNDS`**: Dict mapping problem instances to known bounds/targets
2. **`variables.MAX_RUNTIME`**: Runtime limit in seconds (applies to both bound-based and objective-based problems)
3. **`variables.target_value`**: For objective-based problems

**Note for Objective-Based Problems**: The `compute_objective_value` function should return the raw objective value (e.g., determinant ratio for hadamard matrix problem or sum of radii for circle packing problem) without any additional weighting or combinations. This ensures the results are directly interpretable and comparable.

### Prompt Template Variables:

Use nested access in prompts:
- `{core_parameters.param_name}` for core parameters
- `{target_value}` for target values
- `{PROBLEM_TYPE}` for problem identifier

## Security Considerations

**CRITICAL**: Prevent solution hacking by always recomputing scores:

1. **`count_violations()`**: Must recompute violations from raw `solution_data`, never trust pre-computed counts
2. **`compute_objective_value()`**: Must recompute objective from raw `solution_data`, never trust pre-computed values
3. **Validation**: Always validate solution format and constraints independently

## Testing Your Implementation

Consider adding test cases with known optimal solutions to verify your implementation.

### 1. Manual Testing (make sure it has the same output as running initial_program individually):

```bash
# Test evaluator directly
cd PATH_TO_OPENEVOLVE
OPENEVOLVE_CONFIG_PATH=examples/YOUR_PROBLEM/configs/config_xxx.yaml \
PYTHONPATH=PATH_TO_OPENEVOLVE \
python examples/YOUR_PROBLEM/evaluators/evaluator_modular.py \
examples/YOUR_PROBLEM/initial_programs/initial_program_modular.py

# make sure it has the same output as:
PYTHONPATH=PATH_TO_OPENEVOLVE \
python examples/YOUR_PROBLEM/initial_programs/initial_program_modular.py
```

### 1.5. Testing Verification Functions:

Each `verify.py` should include a `__main__` section for manual testing:

```python
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Solution data file path")
    args = parser.parse_args()
    
    with open(args.input, 'r') as f:
        solution_data = json.load(f)
    
    is_valid, error_msg = validate_solution(solution_data, "test")
    if is_valid:
        # For objective-based problems (e.g., hadamard_matrix, circle_packing)
        objective_value = compute_objective_value(solution_data, "test")
        print(f"{objective_value}")
        # For bound-based problems (e.g., vdw)
        # violations = count_violations(solution_data, "test")
        # print(f"VIOLATIONS: {violations}")
    else:
        print(f"INVALID: {error_msg}")
```

**Note**: The testing approach differs by problem type:
- **Objective-based problems** (Hadamard, Circle Packing): Output the objective value to maximize
- **Bound-based problems** (VDW): Output the violation count to minimize

Test with solution files from evolution runs:
```bash
python examples/YOUR_PROBLEM/evaluators/verify.py --input path/to/solution.json
```

### 2. Expected Output:

- Evaluator should return reasonable scores
- No import errors or configuration warnings
- Initial program should save search results correctly

### 3. Integration Test:

```bash
# Full evolution run
python openevolve-run.py examples/YOUR_PROBLEM/initial_programs/initial_program_modular.py \
  examples/YOUR_PROBLEM/evaluators/evaluator_modular.py \
  --config examples/YOUR_PROBLEM/configs/config_your_problem.yaml \
  --iterations 5
```

## Examples Reference

Study existing implementations:

1. **Bound-Based**: `examples/vdw/` (Van der Waerden numbers)
2. **Objective-Based**: `examples/circle_packing_modular/` (Circle packing)

## Common Pitfalls

1. **Don't add redundant configuration**: Use nested variable access instead of flat variables
2. **Security**: Always recompute scores from raw data
3. **Imports**: Use relative imports within your problem directory
4. **File structure**: Follow the exact directory structure shown above
5. **Template consistency**: Modify only the sections marked as "TODO" or "Problem-specific"

## File Change Summary

When adding a new problem, you need to create these files with the specified change levels:

- **`verify.py`**: 🔴 Completely problem-specific 
- **`initial_program_modular.py`**: 🔴 Completely problem-specific
- **`config_*.yaml`**: 🔴 Completely problem-specific
- **`evaluator_modular.py`**: 🟡 Minimal changes (imports and evaluator type selection)
- **`run.sh`**: 🟡 Minimal changes (names and paths)

Follow this guide exactly to ensure consistency and integration with the OpenEvolve modular framework.