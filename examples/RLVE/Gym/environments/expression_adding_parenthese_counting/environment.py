import random
from typing import Optional
from Gym.environment import VerifiableEnvironment

class Expression_AddingParenthese_Counting_Environment(VerifiableEnvironment):
    prompt_template = \
r"""Given an expression {expression}, please count the number of **distinct values** that can be obtained by inserting parentheses in the expression (but rearranging terms is NOT allowed)."""
    operation_options = ("+", "-", "*")
    
    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the Expression_AddingParenthese_Counting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "num_operands" in self.parameter, "num_operands is required in parameter"
        num_operands = self.parameter["num_operands"]
        assert num_operands >= 3, "num_operands should be greater than or equal to 3"

        operands = self.parameter["operands"] = [random.randint(1, num_operands * num_operands) for _ in range(num_operands)]
        operations = self.parameter["operations"] = [random.choice(self.operation_options) for _ in range(num_operands - 1)]

        dpF = [[set() for _ in range(num_operands)] for _ in range(num_operands)]
        def dp(l, r) -> set :
            if l == r:
                dpF[l][r] = {operands[l]}
                return dpF[l][r]
            if dpF[l][r] :
                return dpF[l][r]
            for i in range(l, r) :
                left_values = dp(l, i)
                right_values = dp(i + 1, r)
                for lv in left_values :
                    for rv in right_values :
                        if operations[i] == "+" :
                            dpF[l][r].add(lv + rv)
                        elif operations[i] == "-" :
                            dpF[l][r].add(lv - rv)
                        elif operations[i] == "*" :
                            dpF[l][r].add(lv * rv)
                        else :
                            raise NotImplementedError(f"Operation {operations[i]} is not implemented")
            return dpF[l][r]
        self.parameter["reference_answer"] = len(dp(0, num_operands - 1))
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(expression = " ".join(str(self.parameter["operands"][i // 2] if i % 2 == 0 else self.parameter["operations"][i // 2]) for i in range(2 * self.parameter["num_operands"] - 1)))
    

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
            if processed_result <= 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]