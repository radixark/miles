"""
Utilities module initialization
"""

from openevolve.utils.async_utils import (
    TaskPool,
    gather_with_concurrency,
    retry_async,
    run_in_executor,
)
from openevolve.utils.code_utils import (
    apply_diff,
    calculate_edit_distance,
    extract_code_language,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)
from openevolve.utils.format_utils import (
    format_metrics_safe,
    format_improvement_safe,
)
from openevolve.utils.metrics_utils import (
    safe_numeric_average,
    safe_numeric_sum,
)
from openevolve.utils.performance_utils import (
    timed_operation,
    timed_async_operation,
)
from openevolve.utils.serialization_utils import (
    save_programs_bulk,
    load_programs_bulk,
    save_programs_individual,
    load_programs_individual,
)

__all__ = [
    "TaskPool",
    "gather_with_concurrency",
    "retry_async",
    "run_in_executor",
    "apply_diff",
    "calculate_edit_distance",
    "extract_code_language",
    "extract_diffs",
    "format_diff_summary",
    "parse_full_rewrite",
    "format_metrics_safe",
    "format_improvement_safe",
    "safe_numeric_average",
    "safe_numeric_sum",
    "timed_operation",
    "timed_async_operation",
    "save_programs_bulk",
    "load_programs_bulk",
    "save_programs_individual",
    "load_programs_individual",
]
