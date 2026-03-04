from __future__ import annotations

import operator as _op
import re
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from functools import lru_cache

import polars as pl


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------


class LabelMatchOp(Enum):
    EQ = "="
    NEQ = "!="
    RE = "=~"


@dataclass
class LabelMatcher:
    label: str
    op: LabelMatchOp
    value: str


@dataclass
class MetricSelector:
    name: str
    matchers: list[LabelMatcher] = field(default_factory=list)


class CompareOp(Enum):
    EQ = "=="
    NEQ = "!="
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="


@dataclass
class CompareExpr:
    selector: MetricSelector
    op: CompareOp
    threshold: float


@dataclass
class RangeFunction:
    func_name: str  # count_over_time, changes, min_over_time, avg_over_time
    selector: MetricSelector
    duration: timedelta


@dataclass
class RangeFunctionCompare:
    func: RangeFunction
    op: CompareOp
    threshold: float


PromQLExpr = MetricSelector | CompareExpr | RangeFunction | RangeFunctionCompare


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"(\d+)([smhd])")
_COMPARE_OPS = sorted(
    [op.value for op in CompareOp], key=len, reverse=True,
)  # longest-first to avoid greedy single-char match
_RANGE_FUNCTIONS = {
    "count_over_time",
    "changes",
    "min_over_time",
    "avg_over_time",
    "max_over_time",
}


def _parse_duration(text: str) -> timedelta:
    match = _DURATION_RE.fullmatch(text)
    if not match:
        raise ValueError(f"Invalid duration: {text!r}")
    value = int(match.group(1))
    unit = match.group(2)
    mapping = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
    return timedelta(**{mapping[unit]: value})


def _parse_label_matchers(text: str) -> list[LabelMatcher]:
    text = text.strip()
    if not text:
        return []

    matchers: list[LabelMatcher] = []
    for part in text.split(","):
        part = part.strip()
        for op_str in ["!=", "=~", "="]:
            if op_str in part:
                label, value = part.split(op_str, 1)
                value = value.strip().strip('"').strip("'")
                matchers.append(LabelMatcher(
                    label=label.strip(),
                    op=LabelMatchOp(op_str),
                    value=value,
                ))
                break

    return matchers


def _parse_metric_selector(text: str) -> MetricSelector:
    text = text.strip()
    if "{" in text:
        name_part, rest = text.split("{", 1)
        labels_part = rest.rstrip("}")
        return MetricSelector(
            name=name_part.strip(),
            matchers=_parse_label_matchers(labels_part),
        )
    return MetricSelector(name=text)


def _find_matching_close(text: str, *, open_ch: str, close_ch: str) -> int:
    depth = 1
    for i, ch in enumerate(text):
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _find_compare_op(text: str) -> tuple[str, CompareOp, str] | None:
    brace_depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
        elif brace_depth == 0:
            for op_str in _COMPARE_OPS:
                if text[i:i + len(op_str)] == op_str:
                    left = text[:i].strip()
                    right = text[i + len(op_str):].strip()
                    if left and right:
                        return left, CompareOp(op_str), right

    return None


@lru_cache(maxsize=256)
def parse_promql(query: str) -> PromQLExpr:
    query = query.strip()

    # Range function: func_name(selector[duration]) [op threshold]
    for func_name in _RANGE_FUNCTIONS:
        prefix = f"{func_name}("
        if query.startswith(prefix):
            inner_and_rest = query[len(prefix):]
            func_end = _find_matching_close(inner_and_rest, open_ch="(", close_ch=")")

            if func_end < 0:
                raise ValueError(f"Unmatched parenthesis in: {query!r}")

            inner = inner_and_rest[:func_end]
            after_func = inner_and_rest[func_end + 1:].strip()

            bracket_match = re.search(r"\[([^\]]+)\]", inner)
            if not bracket_match:
                raise ValueError(f"Missing range selector [duration] in: {query!r}")

            duration = _parse_duration(bracket_match.group(1))
            selector_text = inner[:bracket_match.start()]
            selector = _parse_metric_selector(selector_text)

            range_func = RangeFunction(
                func_name=func_name,
                selector=selector,
                duration=duration,
            )

            if after_func:
                for op_str in _COMPARE_OPS:
                    if after_func.startswith(op_str):
                        threshold_str = after_func[len(op_str):].strip()
                        if threshold_str:
                            return RangeFunctionCompare(
                                func=range_func,
                                op=CompareOp(op_str),
                                threshold=float(threshold_str),
                            )

            return range_func

    # Compare expression: selector op threshold
    compare = _find_compare_op(query)
    if compare:
        left, op, right = compare
        try:
            threshold = float(right)
            return CompareExpr(
                selector=_parse_metric_selector(left),
                op=op,
                threshold=threshold,
            )
        except ValueError:
            pass

    # Plain metric selector
    return _parse_metric_selector(query)


# ---------------------------------------------------------------------------
# Polars comparison helper
# ---------------------------------------------------------------------------


_COMPARE_DISPATCH = {
    CompareOp.EQ: _op.eq,
    CompareOp.NEQ: _op.ne,
    CompareOp.GT: _op.gt,
    CompareOp.LT: _op.lt,
    CompareOp.GTE: _op.ge,
    CompareOp.LTE: _op.le,
}


def compare_col(col: pl.Expr, op: CompareOp, threshold: float) -> pl.Expr:
    fn = _COMPARE_DISPATCH.get(op)
    if fn is None:
        raise ValueError(f"Unknown compare op: {op}")
    return fn(col, threshold)


# ---------------------------------------------------------------------------
# Label matching helper
# ---------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _compile_label_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def match_labels(labels: dict[str, str], matchers: list[LabelMatcher]) -> bool:
    for m in matchers:
        actual = labels.get(m.label, "")
        if m.op == LabelMatchOp.EQ:
            if actual != m.value:
                return False
        elif m.op == LabelMatchOp.NEQ:
            if actual == m.value:
                return False
        elif m.op == LabelMatchOp.RE:
            if not _compile_label_regex(m.value).fullmatch(actual):
                return False
    return True
