"""
Essential Error Constants for OpenEvolve modular_utils
Minimal set of standardized error codes for consistent error handling.
"""

from typing import Dict, Optional
from enum import IntEnum


# TYPES options: "default" or "large_negative"
TYPES = "default"

class ErrorCodes:
    """Essential error codes used by modular_utils"""

    # Core error types ordered by severity (most severe to least severe)
    if TYPES == "default":
        # Most severe: LLM completely failed to understand task
        NO_DIFF_BLOCKS_ERROR = -0.4         # No diff blocks found in LLM response

        # Second most severe: LLM understood task but made no effective changes
        NO_VALID_CHANGE = -0.3              # No valid changes (identical code or lazy output)

        # Third: Execution level failures (at least entered evaluation)
        EXECUTION_ERROR = -0.2              # General execution failures (timeout, no solution, runtime error)

        # Least severe: Validation failures (got solution but invalid)
        VALIDATION_FAILED = -0.1            # Solution failed validation

    else:
        assert False, "[TODO]: Implement error codes for bound based problems"
        
        # # large_negative mode
        # NO_DIFF_BLOCKS_ERROR = -999_999_400  # No diff blocks found in LLM response
        # NO_VALID_CHANGE = -999_999_300       # No valid changes (identical code or lazy output)
        # EXECUTION_ERROR = -999_999_200       # General execution failures
        # VALIDATION_FAILED = -999_999_100     # Solution failed validation

    # Backward compatibility aliases - all currently map to EXECUTION_ERROR
    # Future enhancement: make them fine-grained with unique error codes for better error classification
    EXECUTION_FAILED = EXECUTION_ERROR
    TIMEOUT_OCCURRED = EXECUTION_ERROR
    NO_SOLUTION_FOUND = EXECUTION_ERROR


# Grouped error constants for reward processing
FORMAT_ERRORS = [ErrorCodes.NO_DIFF_BLOCKS_ERROR, ErrorCodes.NO_VALID_CHANGE]
EXECUTION_ERRORS = [ErrorCodes.EXECUTION_ERROR, ErrorCodes.VALIDATION_FAILED]
ALL_ERROR_CODES = FORMAT_ERRORS + EXECUTION_ERRORS


class ErrorThresholds:
    """Error threshold for classification"""

    # Main boundary - anything below this is an error, not normal violations
    if TYPES == "default":
        ERROR_BOUNDARY = -0.1  
    else:
        assert False, "[TODO]: Implement error thresholds for bound based problems"
        # ERROR_BOUNDARY = -999_999_000  

    @classmethod
    def is_error_code(cls, value: int) -> bool:
        """Check if value is an error code (not normal violations)"""
        return value <= cls.ERROR_BOUNDARY


class DefaultValues:
    """Default fallback values for error conditions"""

    if TYPES == "default":
        # Use EXECUTION_ERROR as default fallback (middle severity)
        DEFAULT_ERROR_SCORE = -0.2  # Same as EXECUTION_ERROR
        DEFAULT_KNOWN_BOUND = 1     # Not used in default mode
    else:
        assert False, "[TODO]: Implement default values for bound based problems"
        # # For large negative mode, use multiplier approach
        # DEFAULT_ERROR_SCORE_MULTIPLIER = -1  # Multiply with known_bound
        # DEFAULT_KNOWN_BOUND = 999_999_001    # Default if no known_bound provided


    @classmethod
    def get_error_score(cls, known_bound: Optional[int] = None) -> float:
        """Get standardized error score based on problem's known bound"""
        if TYPES == "default":
            return cls.DEFAULT_ERROR_SCORE
        else:
            assert False, "[TODO]: Implement error score calculation for bound based problems"
            # bound = known_bound if known_bound is not None else cls.DEFAULT_KNOWN_BOUND
            # return cls.DEFAULT_ERROR_SCORE_MULTIPLIER * bound


class ErrorMessages:
    """Human-readable error messages"""

    MESSAGES = {
        ErrorCodes.NO_DIFF_BLOCKS_ERROR: "No diff blocks found in LLM response",
        ErrorCodes.NO_VALID_CHANGE: "No valid changes (identical to existing code or lazy output)",
        ErrorCodes.EXECUTION_ERROR: "Execution failed (timeout, no solution, runtime error, etc.)",
        ErrorCodes.VALIDATION_FAILED: "Solution validation failed",
    }
    
    @classmethod
    def get_message(cls, error_code: int) -> str:
        """Get human-readable message for error code"""
        return cls.MESSAGES.get(error_code, f"Normal violations: {error_code}")


def get_visualization_safe_score(score: float, fallback_value: float = 0.0) -> float:
    """Get visualization-safe version of score, filtering out error codes"""
    if score <= ErrorThresholds.ERROR_BOUNDARY:
        return fallback_value
    return score


if __name__ == "__main__":
    # Simple test
    print("=== Error Constants Test ===")
    test_values = [0, 5, ErrorCodes.EXECUTION_FAILED, ErrorCodes.VALIDATION_FAILED]
    
    for value in test_values:
        print(f"Value {value}: is_error={ErrorThresholds.is_error_code(value)}, "
              f"message='{ErrorMessages.get_message(value)}'")