import math
import sympy
import random
from typing import Optional, List, Dict
from Gym.environment import VerifiableEnvironment
from Gym.environment import timeout, TimeoutException


def generate_test_points(num : int, low : float, high : float) -> List[float] :
    assert num >= 2, "num should be greater than or equal to 2"
    return [low + (high - low) * i / (num - 1) for i in range(num)]


class Integral_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given the derivative of a function: F'(x) = {f_prime}

Your task is to find **an antiderivative** F(x) such that its derivative is equal to the given expression.

**Output Format:** Your answer should be the expression for F(x), written in **SymPy syntax**. Do not omit any symbols (e.g., always use `*` for multiplication).
Example: `sin(2*x)/2` (do **NOT** include quotes or backticks)."""
    test_points = generate_test_points(1024, -2, +2)
    epsilon = 1E-5
    max_val = 1E+4

    def __init__(self,
                 node_type_probs : Optional[List[float]] = None,
                 unary_ops_probs : Dict[str, float] = None,
                 binary_ops_probs : Dict[str, float] = None,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the Integral_Environment instance.
        """
        super().__init__(**kwargs)

        if node_type_probs is None :
            node_type_probs = (0.5, 0.5)
        assert len(node_type_probs) == 2 and abs(sum(node_type_probs) - 1.0) < 1E-8, "node_type_probs should have length 2 and sum to 1"
        self.node_type_probs = node_type_probs

        if unary_ops_probs is None :
            unary_ops_probs = {
                "sin" : 0.1,
                "cos" : 0.1,
                "exp" : 0.05,
                "log" : 0.05,
                "const_pow" : 0.1,
                "const_add" : 0.25,
                "const_mul" : 0.25,
                "const_div" : 0.1,
            }
        assert abs(sum(unary_ops_probs.values()) - 1.0) < 1E-8, "unary_ops_probs values should sum to 1"
        self.unary_ops_probs = unary_ops_probs

        if binary_ops_probs is None :
            binary_ops_probs = {
                "+" : 0.2, 
                "-" : 0.2, 
                "*" : 0.3, 
                "/" : 0.2, 
                "**" : 0.1,
            }
        assert abs(sum(binary_ops_probs.values()) - 1.0) < 1E-8, "binary_ops_probs values should sum to 1"
        self.binary_ops_probs = binary_ops_probs

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }

    def _generate(self) -> None :
        assert "node_num" in self.parameter, "node_num is required in parameter"
        node_num = self.parameter["node_num"]
        assert isinstance(node_num, int) and node_num >= 1, "node_num should be a positive integer"

        unary_ops, unary_probs = zip(*self.unary_ops_probs.items())
        binary_ops, binary_probs = zip(*self.binary_ops_probs.items())

        x = sympy.symbols("x")

        def build_expr(n : int) -> sympy.Expr :
            assert n >= 1, "n should be greater than or equal to 1"
            if n == 1 :
                return x

            if (random.choices(("unary", "binary"), weights = self.node_type_probs, k = 1)[0] if n >= 3 else "unary") == "unary" :
                op = random.choices(unary_ops, weights = unary_probs, k = 1)[0]
                sub = build_expr(n - 1)
                if op == "sin" :
                    return sympy.sin(sub)
                elif op == "cos" :
                    return sympy.cos(sub)
                elif op == "exp" :
                    return sympy.exp(sub)
                elif op == "log" :
                    return sympy.log(sub)
                elif op == "const_pow" :
                    try :
                        if random.random() < 0.5 :
                            return sub ** (1 / sympy.Integer(random.randint(2, 4)))
                        else : # power
                            return sub ** sympy.Integer(random.randint(2, 4))
                    except :
                        # Fall back to a safer option if fractional power fails
                        return sub ** sympy.Integer(random.randint(2, 4))
                elif op == "const_add" :
                    return sub + sympy.Integer(random.choice([-2, -1, +1, +2]))
                elif op == "const_mul" :
                    if random.random() < 0.5 : # negative
                        return sub * -sympy.Integer(random.randint(2, 4))
                    else : # positive
                        return sub * sympy.Integer(random.randint(2, 4))
                elif op == "const_div" :
                    return sub / sympy.Integer(random.randint(2, 4))
                else :
                    raise NotImplementedError(f"Unknown unary op: {op}")
            else :  # binary
                op = random.choices(binary_ops, weights = binary_probs, k = 1)[0]
                assert 1 <= (n - 1) - 1
                left_n = random.randint(1, (n - 1) - 1)
                left = build_expr(left_n)
                right = build_expr((n - 1) - left_n)
                if op == "+" :
                    return left + right
                elif op == "-" :
                    return left - right
                elif op == "*" :
                    return left * right
                elif op == "/" :
                    return left / right
                elif op == "**" :
                    return left ** right
                else :
                    raise NotImplementedError(f"Unknown binary op: {op}")

        while True :
            try :
                f_expr = build_expr(node_num)
                # Add complexity check after building expression
                if sympy.count_ops(f_expr) > 1000:
                    continue
                self.parameter["reference_answer"] = str(f_expr)

                f_prime = sympy.diff(f_expr, x)
                # Add complexity check after differentiation
                if sympy.count_ops(f_prime) > 1000:
                    continue
                self.parameter["f_prime"] = str(f_prime)

                if not f_prime.free_symbols :
                    continue
                if sympy.zoo in f_expr.atoms() or sympy.nan in f_expr.atoms() :
                    continue
                elif sympy.zoo in f_prime.atoms() or sympy.nan in f_prime.atoms() :
                    continue
                else :
                    f_prime_compute = sympy.lambdify(x, f_prime, modules = ["math"])
                    valid_count = 0
                    for pt in self.test_points :
                        try :
                            val = float(f_prime_compute(pt))
                        except :
                            continue
                        if not math.isfinite(val) :
                            continue
                        if abs(val) > self.max_val :
                            valid_count = 0
                            break
                        valid_count += 1
                    if valid_count >= len(self.test_points) // 2 :
                        break
                    else :
                        continue
            except :
                continue


    def _prompt_generate(self) -> str :
        return self.prompt_template.format(f_prime = self.parameter["f_prime"])

    def _process(self, answer : Optional[str]) -> Optional[sympy.Expr] :
        if answer is not None :
            answer = answer.strip()
            # Limit input string length to prevent parsing explosion
            if len(answer) > 10000:
                return None
            try :
                expr = sympy.sympify(answer)
                return expr
            except :
                return None
        else :
            return None

    def scorer(self, output : str) -> float :
        @timeout(10)  # 10 second timeout
        def _scorer_impl():
            processed_result = self.processor(output)
            if processed_result is not None and isinstance(processed_result, sympy.Expr) :
                x = sympy.symbols("x")
                if processed_result.free_symbols - {x} :
                    return self.rewards["wrong_format"]
                
                # Check if processed_result is excessively complex compared to reference
                try :
                    if sympy.count_ops(processed_result) > 4 * sympy.count_ops(sympy.sympify(self.parameter["reference_answer"])) :
                        return self.rewards["wrong_answer"]
                except :
                    return self.rewards["wrong_format"]
                
                try :
                    expr = sympy.diff(processed_result, x) - sympy.sympify(self.parameter["f_prime"])
                    # Add complexity check after differentiation in scorer
                    if sympy.count_ops(expr) > 5000:
                        return self.rewards["wrong_answer"]
                except :
                    return self.rewards["wrong_format"]

                eq = expr.is_zero
                if eq is not None :
                    assert isinstance(eq, bool), "eq should be a boolean value"
                    if eq :
                        return self.rewards["correct_answer"]
                    else :
                        return self.rewards["wrong_answer"]

                try :
                    expr_compute = sympy.lambdify(x, expr, modules = ["math"])
                except :
                    return self.rewards["wrong_answer"]
                zero_count = 0
                for pt in self.test_points :
                    try :
                        val = float(expr_compute(pt))
                    except :
                        continue
                    if not math.isfinite(val) :
                        continue
                    if abs(val) > self.epsilon :
                        return self.rewards["wrong_answer"]
                    else :
                        zero_count += 1

                if zero_count >= len(self.test_points) // 4 :
                    return self.rewards["correct_answer"]
                else :
                    return self.rewards["wrong_answer"]
            else :
                return self.rewards["wrong_format"]
        
        try:
            return _scorer_impl()
        except TimeoutException:  # Catch the specific timeout exception
            return -1.0