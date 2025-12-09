import random
from typing import Optional, List, Dict
from Gym.environment import VerifiableEnvironment


class Circuit_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""There are {N} boolean (0/1) values x[0], x[1], ..., x[{N_minus_1}].

Given a Boolean expression (where `&` is bitwise AND, `|` is bitwise OR, and `^` is bitwise XOR): {expression}
Please find any solution x[0], x[1], ..., x[{N_minus_1}] that makes the expression evaluate to 1.

Output Format: Your final answer should be a single line containing x[0], x[1], ..., x[{N_minus_1}], separated by **spaces**.
Example: `{N_boolean}` (do **NOT** include quotes or backticks)."""

    def __init__(self,
                 binary_ops_probs : Dict[str, float] = None,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, correct_solution : float = +1.0, wrong_solution : float = 0.0,
                 **kwargs) :
        """
        Initialize the Circuit_Environment instance.
        """
        super().__init__(**kwargs)

        if binary_ops_probs is None :
            binary_ops_probs = {
                "&" : 0.25, 
                "|" : 0.25, 
                "^" : 0.5,
            }
        assert abs(sum(binary_ops_probs.values()) - 1.0) < 1E-8, "binary_ops_probs values should sum to 1"
        self.binary_ops_probs = binary_ops_probs

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "correct_solution" : correct_solution,
            "wrong_solution" : wrong_solution,
        }
    
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert M >= N, "M should be greater than or equal to N"

        binary_ops, binary_probs = zip(*self.binary_ops_probs.items())

        while True :
            x = [random.randint(0, 1) for i in range(N)]

            def build_tree(n) :
                if n == 1 :
                    index = random.randint(0, N - 1)
                    return index, x[index]
                left_n = random.randint(1, n - 1)
                right_n = n - left_n
                left_tree, left_value = build_tree(left_n)
                right_tree, right_value = build_tree(right_n)
                op = random.choices(binary_ops, weights = binary_probs, k = 1)[0]
                if op == "&" :
                    value = left_value & right_value
                elif op == "|" :
                    value = left_value | right_value
                elif op == "^" :
                    value = left_value ^ right_value
                else :
                    raise ValueError("Invalid operator")
                return (left_tree, op, right_tree), value
            tree, value = build_tree(M)
            
            if value == 1 :
                self.parameter["reference_answer"] = " ".join(map(str, x))
                self.parameter["tree"] = tree
                break
    
    def build_expression(self, tree) :
        if isinstance(tree, int) :
            return "x[{}]".format(tree)
        left_tree, op, right_tree = tree
        return "({} {} {})".format(self.build_expression(left_tree), op, self.build_expression(right_tree))
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            expression = self.build_expression(self.parameter["tree"])[1 : -1],
            N_boolean = " ".join(str(i % 2) for i in range(self.parameter["N"])),
        )

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            x = processed_result
            if len(x) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not all(xi in (0, 1) for xi in x) :
                return self.rewards["invalid_solution"]
            
            def compute(tree) :
                if isinstance(tree, int) :
                    return x[tree]
                left_tree, op, right_tree = tree
                left_value = compute(left_tree)
                right_value = compute(right_tree)
                if op == "&" :
                    return left_value & right_value
                elif op == "|" :
                    return left_value | right_value
                elif op == "^" :
                    return left_value ^ right_value
                else :
                    raise ValueError("Invalid operator")
            
            if compute(self.parameter["tree"]) == 1 :
                return self.rewards["correct_solution"]
            else :
                return self.rewards["wrong_solution"]
        else :
            return self.rewards["wrong_format"]