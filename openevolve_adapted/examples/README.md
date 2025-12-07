# Examples

Collection of optimization problem domains for evolutionary algorithm experimentation. Some of them (like circle packing) are refacted from OpenEvolve.

## Available Problems

- **circle_packing_modular**: Circle packing optimization
- **first_autocorr_inequality**: First autocorrelation inequality
- **second_autocorr_inequality**: Second autocorrelation inequality
- **third_autocorr_inequality**: Third autocorrelation inequality
- **hadamard_matrix**: Hadamard matrix optimization

## Problem Directory Structure

```
PROBLEM_NAME/
├── configs/
│   └── config_PROBLEM_NAME_*.yaml       # Evolution parameters (population, MAP-Elites, etc.)
├── evaluators/
│   ├── evaluator_modular.py             # Standard evaluation harness
│   └── verify.py                        # Problem-specific scoring
└── initial_programs/
    ├── initial_program.py               # Baseline solution
    └── ref/                             # (Optional) Reference data/SOTA solutions
```

## File Descriptions

**verify.py**: Problem-specific evaluation logic

```python
def compute_objective_value(solution_data) -> float:
    """Return scalar score to maximize"""
    return score
```

**evaluator_modular.py**: Standard harness that imports and executes `verify.py`. Use as template—minimal changes needed.

**initial_program.py**: Baseline solution serving as evolution starting point.

**config_*.yaml**: Database size, MAP-Elites feature dimensions, reinitialization settings.

**ref/**: Optional reference implementations (SOTA, utilities). Accessed by evolved programs via:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ref"))
from ref_module import function
```

## Adding a New Problem

1. Create directory structure with `configs/`, `evaluators/`, `initial_programs/`

2. Implement `verify.py`:
```python
def compute_objective_value(solution_data) -> float:
    # Return objective score
    return float(score)
```

3. Copy `evaluator_modular.py` from existing problem (mostly standard—only import from problem-specific `verify.py`)

4. Create `initial_program.py` with baseline solution

5. Create `config_PROBLEM_NAME.yaml` with evolution parameters

## Debugging

Test a problem standalone using `run_init_example.sh`:

```bash
# Edit run_init_example.sh in openevolve_adapted and run:
bash run_init_example.sh
```

This validates evaluator setup, scoring logic, and evolution without training.
