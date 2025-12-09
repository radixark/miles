import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class BinaryLinearEquation_SolutionCounting_Environment(VerifiableEnvironment) :
    prompt_template = r"""What is the number of integer solution pairs (x, y) such that ({A}) * x + ({B}) * y + ({C}) = 0, with {X1} <= x <= {X2} and {Y1} <= y <= {Y2}?"""

    def __init__(self,
                wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                not_guaranteed_probability : float = 0.05,
                **kwargs) :
        """
        Initialize the BinaryLinearEquation_SolutionCounting instance.
        """
        super().__init__(**kwargs)

        self.not_guaranteed_probability = not_guaranteed_probability
        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_RANGE" in self.parameter, "MAX_RANGE is required in parameter"
        MAX_RANGE = self.parameter["MAX_RANGE"]
        assert MAX_RANGE >= 8, "MAX_RANGE must be at least 8"

        A = self.parameter["A"] = random.randint(-MAX_RANGE, +MAX_RANGE)
        B = self.parameter["B"] = random.randint(-MAX_RANGE, +MAX_RANGE)
        not_guaranteed = random.random() < self.not_guaranteed_probability
        if not_guaranteed :
            X1 = self.parameter["X1"] = random.randint(-MAX_RANGE, +MAX_RANGE)
            X2 = self.parameter["X2"] = random.randint(X1, +MAX_RANGE)
            Y1 = self.parameter["Y1"] = random.randint(-MAX_RANGE, +MAX_RANGE)
            Y2 = self.parameter["Y2"] = random.randint(Y1, +MAX_RANGE)
            C = self.parameter["C"] = random.randint(-2 * (MAX_RANGE ** 2),+2 * (MAX_RANGE ** 2))
        else :
            x = random.randint(-MAX_RANGE, +MAX_RANGE)
            y = random.randint(-MAX_RANGE, +MAX_RANGE)
            C = self.parameter["C"] = -(A * x + B * y)
            X1 = self.parameter["X1"] = random.randint(-MAX_RANGE, x)
            X2 = self.parameter["X2"] = random.randint(x, +MAX_RANGE)
            Y1 = self.parameter["Y1"] = random.randint(-MAX_RANGE, y)
            Y2 = self.parameter["Y2"] = random.randint(y, +MAX_RANGE)
        

        def gcd(a, b):
            while b:
                a, b = b, a % b
            return abs(a)

        def extended_gcd_positive(a, b):
            # Returns (g, x, y) with a*x + b*y = g, for a,b >= 0
            if b == 0:
                return (a, 1, 0)
            g, x1, y1 = extended_gcd_positive(b, a % b)
            return (g, y1, x1 - (a // b) * y1)

        def ceil_div(a, b):
            # Ceil division that works for any sign of b
            return -((-a) // b)

        def floor_div(a, b):
            # Floor division (Python's // already floors)
            return a // b

        def k_range(a0, step, L, R):
            """
            From constraint: L <= a0 + step*k <= R
            Return [lo, hi] for integer k, or (1, 0) for empty.
            """
            if step > 0:
                lo = ceil_div(L - a0, step)
                hi = floor_div(R - a0, step)
            else:  # step < 0
                # Inequality reverses when dividing by a negative
                lo = ceil_div(R - a0, step)
                hi = floor_div(L - a0, step)
            return lo, hi

        def compute(A, B, C, X1, X2, Y1, Y2):
            if X1 > X2:
                X1, X2 = X2, X1
            if Y1 > Y2:
                Y1, Y2 = Y2, Y1

            # Degenerate cases
            if A == 0 and B == 0:
                return (X2 - X1 + 1) * (Y2 - Y1 + 1) if C == 0 else 0

            if A == 0:
                # B*y + C = 0
                if C % B == 0:
                    y = -C // B
                    return (X2 - X1 + 1) if (Y1 <= y <= Y2) else 0
                else:
                    return 0

            if B == 0:
                # A*x + C = 0
                if C % A == 0:
                    x = -C // A
                    return (Y2 - Y1 + 1) if (X1 <= x <= X2) else 0
                else:
                    return 0

            # General case
            d = gcd(A, B)
            if C % d != 0:
                return 0

            # Find one solution to A*x + B*y = -C
            _, xg, yg = extended_gcd_positive(abs(A), abs(B))  # gives axg + byg = gcd(|A|,|B|)
            if A < 0:
                xg = -xg
            if B < 0:
                yg = -yg

            mult = (-C) // d
            x0 = xg * mult
            y0 = yg * mult

            # Parametric form
            step_x = B // d
            step_y = -A // d  # note: can be negative

            # k-range from x and y intervals
            kx_lo, kx_hi = k_range(x0, step_x, X1, X2)
            ky_lo, ky_hi = k_range(y0, step_y, Y1, Y2)

            lo = max(kx_lo, ky_lo)
            hi = min(kx_hi, ky_hi)

            return 0 if lo > hi else hi - lo + 1

        self.parameter["reference_answer"] = compute(A, B, C, X1, X2, Y1, Y2)
        if not not_guaranteed :
            assert self.parameter["reference_answer"] >= 1
        else :
            assert self.parameter["reference_answer"] >= 0
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            A = self.parameter["A"],
            B = self.parameter["B"],
            C = self.parameter["C"],
            X1 = self.parameter["X1"],
            X2 = self.parameter["X2"],
            Y1 = self.parameter["Y1"],
            Y2 = self.parameter["Y2"],
        )
    

    def _process(self, answer : Optional[str]) -> Optional[int] :
        if answer is not None :
            answer = answer.strip()
            try :
                int_answer = int(answer)
                return int_answer
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result < 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                if self.parameter["reference_answer"] == 0 :
                    return self.rewards["rewarding_weight"] * (processed_result == 0)
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]