"""
RLVE Reward Model - Verifies answers using RLVE Gym verifiable environments.

This module provides the reward computation for RLVE training by:
1. Selecting the appropriate answer markers based on configuration
2. Instantiating the environment with the problem config
3. Running the verifier on the model's response
"""
from typing import Any, Dict

# RLVE Gym imports - must be in PYTHONPATH
try:
    from Gym.environment import VerifiableEnvironment
    from Gym.environments import identifier2environment
    RLVE_AVAILABLE = True
except ImportError:
    RLVE_AVAILABLE = False


def rlve_rm(args, environment: str, config: Dict, response: str) -> Dict[str, Any]:
    """
    Compute reward for RLVE training using verifiable environments.

    Args:
        args: Training arguments containing answer_marker_type and custom_prompt_preprocessor
        environment: Name of the environment (e.g., "Sorting", "Fibonacci")
        config: Problem configuration from the environment generator
        response: Model's response string

    Returns:
        Dict with keys: reward, accuracy, format_score (from environment verifier)
    """
    if not RLVE_AVAILABLE:
        raise ImportError(
            "RLVE Gym not available. Ensure Gym package is in PYTHONPATH. "
            "Example: export PYTHONPATH=/path/to/RLVE:$PYTHONPATH"
        )

    # Select answer markers based on configuration
    if args.answer_marker_type == r"\boxed{}":
        answer_markers = (r"\boxed{", r"}")
        # Validate preprocessor compatibility
        if hasattr(args, 'custom_prompt_preprocessor') and args.custom_prompt_preprocessor is not None:
            assert args.custom_prompt_preprocessor in ("ChatTemplate_NoSystemPrompt",), \
                f"\\boxed{{}} marker requires ChatTemplate_NoSystemPrompt preprocessor, got {args.custom_prompt_preprocessor}"
    elif args.answer_marker_type == r"<answer></answer>":
        answer_markers = (r"<answer>", r"</answer>")
        # Validate preprocessor compatibility
        if hasattr(args, 'custom_prompt_preprocessor') and args.custom_prompt_preprocessor is not None:
            assert args.custom_prompt_preprocessor in ("TinyZero",), \
                f"<answer></answer> marker requires TinyZero preprocessor, got {args.custom_prompt_preprocessor}"
    else:
        raise NotImplementedError(f"Answer marker type {args.answer_marker_type} not implemented.")

    # Create environment instance with answer markers
    problem: VerifiableEnvironment = identifier2environment[environment](answer_markers=answer_markers)

    # Set the problem configuration (this restores the problem state)
    problem.set_config(config)

    # Run the verifier and return the result
    # The verifier returns a dict with: reward, accuracy, format_score
    return problem.verifier(response)
