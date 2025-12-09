import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Josephus_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1996
    prompt_template = \
r"""{N} people are standing in a circle (labeled from 1 to {N}). Starting from the person labeled 1, they count off in order. The person who counts to {M} is eliminated, and the next person resumes counting from 1. This process continues until everyone is eliminated. Please determine the order in which people are eliminated.

**Output Format:** Your final answer should be a single line containing the labels of the people in the order they are eliminated, separated by spaces."""
    
    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the Josephus_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 3, "MAX_N should be greater than or equal to 3"

        N = self.parameter["N"] = random.randint(3, MAX_N)
        M = self.parameter["M"] = random.randint(2, N)


        bit = [0] * (N + 1)

        def lowbit(x) :
            return x & -x

        def add(pos, val) :
            while pos <= N :
                bit[pos] += val
                pos += lowbit(pos)

        def find_kth(k) :
            idx = 0
            curr = 0
            max_bit = N.bit_length()
            for i in range(max_bit, -1, -1) :
                next_idx = idx + (1 << i)
                if next_idx <= N and curr + bit[next_idx] < k:
                    idx = next_idx
                    curr += bit[next_idx]
            return idx + 1

        for i in range(1, N + 1) :
            add(i, 1)

        result = []
        remaining = N
        cur = 1
        for _ in range(N) :
            cur = (cur - 1 + M - 1) % remaining + 1
            person = find_kth(cur)
            result.append(person)
            add(person, -1)
            remaining -= 1

        self.parameter["gold_answer"] = result
        assert len(result) == N, "The length of the result should be equal to N"
        self.parameter["reference_answer"] = " ".join(map(str, result))
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], M = self.parameter["M"])


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
                return self.rewards["invalid_answer"]
            if len(set(processed_result)) != self.parameter["N"] :
                return self.rewards["invalid_answer"]
            if not all(1 <= x <= self.parameter["N"] for x in processed_result) :
                return self.rewards["invalid_answer"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(float(a == b) for a, b in zip(self.parameter["gold_answer"], processed_result)) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]