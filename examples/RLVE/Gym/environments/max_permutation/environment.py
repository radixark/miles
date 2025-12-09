import random
from functools import cmp_to_key
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MaxPermutation_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1018
    prompt_template = \
r"""You are given an array `A` of {N} positive integers:
{array}

Your task is to rearrange **all** the elements of the array (each number must be used **exactly once**) to form the **largest possible integer** when the numbers are **concatenated in order**. Treat the numbers as **strings** during concatenation (not as digits or arithmetic values).


Output Format:
Your final answer should be a **single line** with the indices of the chosen arrangement, separated by **spaces**.
Example: `{ALL_INDICES}` (do **NOT** include the backticks or quotes) means the numbers are used in the order: {ALL_ITEMS}.
"""

    def __init__(self,
                 proportion_being_prefix : float = 0.6,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MaxPermutation_Environment instance.
        
        Args:
            proportion_being_prefix (float): Proportion of the numbers in the array that are prefixes of other numbers.
        """
        super().__init__(**kwargs)

        assert 0.0 <= proportion_being_prefix < 1.0, "proportion_being_prefix should be in [0.0, 1.0)"
        self.proportion_being_prefix = proportion_being_prefix

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        assert "MAX_DIGIT_NUM" in self.parameter, "MAX_DIGIT_NUM is required in parameter"
        MAX_DIGIT_NUM = self.parameter["MAX_DIGIT_NUM"]
        assert MAX_DIGIT_NUM >= 1, "MAX_DIGIT_NUM should be greater than or equal to 1"

        M = N - int(N * self.proportion_being_prefix)
        assert M >= 1, "M should be greater than or equal to 1"
        array = self.parameter["array"] = ["".join([str(random.randint(1, 2)) for _ in range(MAX_DIGIT_NUM)]) for i in range(M)]
        for i in range(N - M) :
            prefix = random.choice(array[: M])
            assert len(prefix) == MAX_DIGIT_NUM, "prefix should have the same length as MAX_DIGIT_NUM"
            array.append(prefix[: random.randint(1, MAX_DIGIT_NUM)])
        random.shuffle(array)
        
        # Sort the array in descending order based on concat(a + b) > (b + a)
        def cmp(a : dict, b : dict) -> int :
            a, b = a["value"], b["value"]
            if a + b > b + a :
                return -1
            elif a + b < b + a :
                return 1
            else :
                return 0
        self.parameter["reference_answer"] = [dict(index = i, value = a) for i, a in enumerate(array)]
        self.parameter["reference_answer"].sort(key = cmp_to_key(cmp))
        self.parameter["gold"] = int("".join([item["value"] for item in self.parameter["reference_answer"]]))
        self.parameter["reference_answer"] = " ".join([str(item["index"]) for item in self.parameter["reference_answer"]])
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            array = "\n".join("A[{}]={}".format(i, a) for i, a in enumerate(self.parameter["array"])),
            ALL_INDICES = " ".join(str(i) for i in range(self.parameter["N"] - 1, -1, -1)),
            ALL_ITEMS = ", ".join("A[{}]".format(i) for i in range(self.parameter["N"] - 1, -1, -1)),
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
                return self.rewards["invalid_solution"]
            if len(set(processed_result)) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= i < self.parameter["N"] for i in processed_result) :
                return self.rewards["invalid_solution"]
            
            answer = int("".join([self.parameter["array"][i] for i in processed_result]))
            assert answer <= self.parameter["gold"], "answer should be less than or equal to gold"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / self.parameter["gold"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == self.parameter["gold"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]