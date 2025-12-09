import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class BitEquationCounting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Given a Boolean expression (where `_` represents a variable that can be 0 or 1, `&` is bitwise AND, `|` is bitwise OR, and `^` is bitwise XOR): {expression}

There are 2^{N} possible combinations of values for the variables. Your task is to find how many of these combinations make the expression evaluate to true.

**Output Format:** Your final answer should be a single integer — the number of combinations that make the expression true. Example: `15` (do **NOT** include quotes or backticks)."""

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the BitEquationCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"\
        
        def build_expression(n) :
            if n == 1 :
                return "_", 1, 1
            left_n = random.randint(1, n - 1)
            right_n = n - left_n
            left_expr, left_true, left_false = build_expression(left_n)
            right_expr, right_true, right_false = build_expression(right_n)
            op = random.choice(("&", "|", "^"))
            if op == "&" :
                true_count = left_true * right_true
                false_count = (2 ** n) - true_count
            elif op == "|" :
                false_count = left_false * right_false
                true_count = (2 ** n) - false_count
            elif op == "^" :
                true_count = left_true * right_false + left_false * right_true
                false_count = left_true * right_true + left_false * right_false
                assert true_count + false_count == 2 ** n, "XOR operation should cover all cases"
            else :
                raise ValueError("Invalid operator")
            return "({} {} {})".format(left_expr, op, right_expr), true_count, false_count
        expression, true_count, false_count = build_expression(N)

        self.parameter["expression"] = expression[1 : -1]
        self.parameter["reference_answer"] = true_count
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(expression = self.parameter["expression"], N = self.parameter["N"])
    

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
            if not (0 <= processed_result <= 2 ** self.parameter["N"]) :
                return self.rewards["wrong_range"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]