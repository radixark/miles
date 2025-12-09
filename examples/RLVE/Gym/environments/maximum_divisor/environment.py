import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MaximumDivisor_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2440
    prompt_template = \
r"""You are given an array A of length {N}. The values are as follows (indexing starts at 0):
{A}

Please find the **maximum positive integer L** such that the following inequality holds: [A[0] / L] + [A[1] / L] + ... + [A[{N_minus_1}] / L] >= {K}, where [x] denotes the **floor function** (i.e., rounding down to the nearest integer).

**Output Format:**
Your final answer should be a single line containing the value of L.
"""


    def __init__(self,
                 random_range_coefficient : int = 20,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = +5.0,
                 **kwargs) :
        """
        Initialize the MaximumDivisor_Environment instance.
        """
        super().__init__(**kwargs)

        self.random_range_coefficient = random_range_coefficient

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
        assert N >= 2, "N must be at least 2"

        K = self.parameter["K"] = random.randint(1, N * max(1, N // self.random_range_coefficient))

        A = self.parameter["A"] = [random.randint(1, N) for i in range(N)]

        if sum(A) < K :
            A[0] += K - sum(A)
        assert sum(A) >= K, "sum(A) must be at least K"
        random.shuffle(A)


        def check(l) :
            return sum(li // l for li in A) >= K

        l, r = 1, max(A) + 1
        while l < r :
            m = (l + r) // 2
            if check(m) :
                l = m + 1
            else :
                r = m
        self.parameter["reference_answer"] = l - 1
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            A = " ".join(map(str, self.parameter["A"])),
            K = self.parameter["K"],
        )


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
            
            if sum(li // processed_result for li in self.parameter["A"]) >= self.parameter["K"] :
                assert processed_result <= self.parameter["reference_answer"]

                if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                    return self.rewards["rewarding_weight"] * ((processed_result / self.parameter["reference_answer"]) ** self.rewards["rewarding_beta"])
                elif self.rewards["rewarding_strategy"] == "gold=answer" :
                    return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == processed_result)
                else :
                    raise ValueError("Invalid rewarding strategy")
            else :
                return self.rewards["invalid_solution"]
        else :
            return self.rewards["wrong_format"]