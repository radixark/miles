from __future__ import annotations


def exact_div(numerator: int, denominator: int) -> int:
    assert denominator != 0, f"cannot divide {numerator} by zero"
    assert numerator % denominator == 0, f"{numerator} is not a whole number of {denominator}"
    return numerator // denominator
