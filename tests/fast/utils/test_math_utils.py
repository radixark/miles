import pytest

from miles.utils.math_utils import exact_div


class TestExactDiv:
    def test_divides_when_the_division_comes_out_whole(self):
        """The only case a caller wants: a count that tiles the divisor exactly."""
        assert exact_div(24, 8) == 3

    def test_refuses_a_remainder(self):
        """Flooring a remainder away is what this helper exists to prevent."""
        with pytest.raises(AssertionError, match="12 is not a whole number of 8"):
            exact_div(12, 8)

    def test_refuses_a_zero_divisor(self):
        """A caller that forgot to fill in the divisor must hear about it, not see a ZeroDivisionError."""
        with pytest.raises(AssertionError, match="divide 8 by zero"):
            exact_div(8, 0)
