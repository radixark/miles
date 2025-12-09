import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class IntegerProgramming_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""There are {N} integers x[0], x[1], ..., x[{N_minus_1}]. They satisfy the following {M} inequations:
{inequations}

Please find any solution x[0], x[1], ..., x[{N_minus_1}] that satisfies the inequations.

Output Format: Your final answer should be a single line containing x[0], x[1], ..., x[{N_minus_1}], separated by **spaces**.
Example: `{one_to_N}` (do **NOT** include quotes or backticks); this means: x[0] = 1, x[1] = 2, ..., x[{N_minus_1}] = {N}.
"""

    def __init__(self,
                 number_range : int = 4,
                 coefficient_non_zero_probability : float = 0.5,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        super().__init__(**kwargs)

        self.number_range = number_range

        self.coefficient_non_zero_probability = coefficient_non_zero_probability

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert M >= 1, "M should be greater than or equal to 1"

        self.parameter["x"] = [random.randint(-N, +N) for i in range(N)]
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["x"]))

        inequations = self.parameter["inequations"] = []
        results = self.parameter["results"] = []
        for m in range(M) :
            while True :
                inequation = []
                for i in range(N) :
                    if random.random() < self.coefficient_non_zero_probability :
                        coefficient = random.randint(1, self.number_range)
                        if random.random() < 0.5 :
                            coefficient = -coefficient
                    else :
                        coefficient = 0
                    inequation.append(coefficient)
                if any(inequation) :
                    break
            inequations.append(inequation) # left >= right
            results.append(sum(coefficient * xi for coefficient, xi in zip(inequation, self.parameter["x"])) - random.randint(0, self.number_range // 2))
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            N_minus_1 = self.parameter["N"] - 1,
            M = self.parameter["M"],
            inequations = "\n".join(" + ".join("({}) * x[{}]".format(coefficient, i) for i, coefficient in enumerate(inequation) if coefficient != 0) + " >= {}".format(result) for inequation, result in zip(self.parameter["inequations"], self.parameter["results"])),
            one_to_N = " ".join(map(str, range(1, self.parameter["N"] + 1))),
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
                return self.rewards["wrong_format"]
            
            satisfied = sum(int(sum(coefficient * xi for coefficient, xi in zip(inequation, x)) >= result) for inequation, result in zip(self.parameter["inequations"], self.parameter["results"]))
            
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / len(self.parameter["inequations"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == len(self.parameter["inequations"]))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]