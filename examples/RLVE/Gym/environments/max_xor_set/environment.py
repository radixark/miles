import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MaxXorSet_Environment(VerifiableEnvironment):  # Source: https://www.luogu.com.cn/problem/P3812
    prompt_template = \
r"""You are given an array A of {N} positive integers:
{array}

Please select indices i_1, ..., i_k (k is arbitrary) to maximize A[i_1] XOR ... XOR A[i_k] (i.e., the bitwise XOR of the selected elements).

Output Format: Your final answer should be a **single line** containing i_1, ..., i_k (the indices of the selected elements), separated by **spaces**."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MaxXorSet_Environment instance.
        """
        super().__init__(**kwargs)

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

        assert "MAX_bit_length" in self.parameter, "MAX_bit_length is required in parameter"
        MAX_bit_length = self.parameter["MAX_bit_length"]
        assert MAX_bit_length >= 2, "MAX_bit_length should be greater than or equal to 2"

        A = self.parameter["A"] = [random.randint(1, 2 ** MAX_bit_length - 2) for _ in range(N)]


        max_value = max(A)
        max_bit_index = max_value.bit_length() - 1  # if max_value is 0, this will be -1

        # Initialize the basis array P with size = max_bit_index+1
        # If max_bit_index == -1 (all A are zero), P will be an empty list.
        P = [0] * (max_bit_index + 1)

        def insert(x):
            k = x
            # Insert k into the basis
            for i in range(max_bit_index, -1, -1):
                if not (k >> i) & 1:
                    continue
                if P[i] == 0:
                    P[i] = k
                    return
                k ^= P[i]

        def max_xor():
            res = 0
            for i in range(max_bit_index, -1, -1):
                res = max(res, res ^ P[i])
            return res

        # Build the basis
        for x in A:
            insert(x)

        self.parameter["gold_answer"] = max_xor()
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            array = "\n".join("A[{}]={}".format(i, a) for i, a in enumerate(self.parameter["A"])),
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

            if not all(0 <= i < self.parameter["N"] for i in processed_result) :
                return self.rewards["invalid_solution"]
            
            answer = 0
            for i in processed_result :
                answer ^= self.parameter["A"][i]

            assert answer <= self.parameter["gold_answer"], "answer should be less than or equal to gold"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / self.parameter["gold_answer"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == self.parameter["gold_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]