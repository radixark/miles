import math
import sympy
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class PolynomialMinimum_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Given f(x) = {polynomial}, find the value of x0 that minimizes f(x). Your final answer should be a single real number in decimal form, representing the value of x0."""

    def __init__(self,
                 max_weight : int = 2,
                 wrong_format : float = -1.0, rewarding_strategy : str = "piecewise", rewarding_threshold : float = +0.95, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the PolynomialMinimum_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_weight = max_weight

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_threshold" : rewarding_threshold,
            "rewarding_beta" : rewarding_beta,
        }

        if self.rewards["rewarding_strategy"] == "piecewise" :
            self.passing_reward_threshold = rewarding_threshold * (0.999 ** rewarding_beta)
        else :
            raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2 and N % 2 == 0, "N should be greater than or equal to 2 and even"

        available_degrees = list(range(2, N, 2))
        random.shuffle(available_degrees)

        degrees = [N] + available_degrees

        x = sympy.Symbol("x")
        terms = []
        for deg in degrees :
            a = random.randint(1, self.max_weight)
            s = random.choice(range(-self.max_weight, +self.max_weight + 1))
            term = a * ((x - s) ** deg)
            terms.append(term)

        poly = sum(terms)
        poly_expanded = sympy.expand(poly)
        coeffs = [int(poly_expanded.coeff(x, i)) for i in range(N + 1)]

        assert len(coeffs) == N + 1, "coeffs should have length N + 1"
        assert coeffs[N] > 0.0, "leading coefficient should be positive"
        self.parameter["coeffs"] = coeffs


        f_expr = sum(c * (x ** i) for i, c in enumerate(coeffs))
        real_roots = [0.0] + [random.uniform(-self.max_weight, self.max_weight) for _ in range(5)]
        try :
            # (Try to) Find the minimum of the polynomial using sympy
            d_expr = sympy.diff(f_expr, x)
            roots = sympy.nroots(d_expr)
            real_roots += [float(sympy.re(r)) for r in roots if abs(sympy.im(r)) < 1E-6]
        except :
            pass
        f_vals = [float(f_expr.evalf(subs = {x : xr})) for xr in real_roots]
        min_idx = f_vals.index(min(f_vals))
        x0 = real_roots[min_idx]
        self.parameter["reference_answer"] = float(x0)
        self.parameter["reference_value"] = float(f_vals[min_idx])
        self.parameter["worst_value"] = f_vals[0]
    
    def _prompt_generate(self) -> str :
        x = sympy.Symbol("x")
        poly_expr = sum(c * (x ** i) for i, c in enumerate(self.parameter["coeffs"]))
        return self.prompt_template.format(polynomial = sympy.simplify(poly_expr))


    def _process(self, answer : Optional[str]) -> Optional[float] :
        if answer is not None :
            answer = answer.strip()
            try :
                float_answer = float(answer)
                if not math.isfinite(float_answer) :
                    return None
                return float_answer
            except ValueError :
                return None
        else :
            return None
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            def compute(x_val : float) -> float :
                x = sympy.Symbol("x")
                f_expr = sum(c * (x ** i) for i, c in enumerate(self.parameter["coeffs"]))
                return float(f_expr.evalf(subs = {x : x_val}))
            f_val = compute(processed_result)

            if self.rewards["rewarding_strategy"] == "piecewise" :
                if f_val >= self.parameter["worst_value"] :
                    return self.rewards["rewarding_threshold"] * (f_val <= self.parameter["reference_value"])
                elif f_val >= self.parameter["reference_value"] :
                    # self.parameter["reference_value"] <= f_val < self.parameter["worst_value"]
                    return self.rewards["rewarding_threshold"] * (((self.parameter["worst_value"] - f_val) / (self.parameter["worst_value"] - self.parameter["reference_value"])) ** self.rewards["rewarding_beta"])
                else :
                    # f_val < self.parameter["reference_value"]
                    return self.rewards["rewarding_threshold"] + (1.0 - self.rewards["rewarding_threshold"]) / (1 + 1 / max(self.parameter["reference_value"] - f_val, 1E-8))
        else :
            return self.rewards["wrong_format"]