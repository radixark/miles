import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class PolynomialFactorization_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a degree-{N} polynomial: (x - a_1)...(x - a_{N}) = {polynomial}

Your task is to find any valid set of integers `a_1, ..., a_{N}` (not necessarily distinct) such that the product of the linear factors on the left expands to match the given polynomial.

**Output Format:** Your final answer should be a single line containing `a_1, ..., a_{N}`, separated by **spaces**."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the PolynomialFactorization instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        self.parameter["gold_answer"] = [random.randint(-N, +N) for _ in range(N)]
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["gold_answer"]))

        coefficients = self.parameter["coefficients"] = [1] + [0] * N
        for a in self.parameter["gold_answer"] :
            for i in range(N, 0, -1) :
                coefficients[i] = coefficients[i - 1] - a * coefficients[i]
            coefficients[0] *= -a
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            polynomial = " + ".join("({}) * x^{}".format(coefficient, i) for i, coefficient in enumerate(self.parameter["coefficients"]) if coefficient != 0),
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

            if len(processed_result) != self.parameter["N"] :
                return self.rewards["wrong_format"]
            
            # make a multiset of self.parameter["gold_answer"]
            gold_answer_multiset = {}
            for a in self.parameter["gold_answer"] :
                if a in gold_answer_multiset :
                    gold_answer_multiset[a] += 1
                else :
                    gold_answer_multiset[a] = 1
            
            satisfied = 0
            for a in processed_result :
                if gold_answer_multiset.get(a, 0) > 0 :
                    satisfied += 1
                    gold_answer_multiset[a] -= 1
            
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == self.parameter["N"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]