import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class NumberPartitionCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1025
    prompt_template = \
r"""You are given a positive integer {N}. Your task is to divide it into exactly {K} **non-empty** positive integers such that:

- The **sum** of the {K} parts is exactly {N},
- The **order does not matter** — that is, two partitions are considered the same if they contain the same numbers, regardless of order (e.g., `1 + 1 + 5` is the same as `5 + 1 + 1`),
- All parts must be strictly positive integers (no zero).

Determine how many **distinct** ways there are to partition the number {N} into {K} such parts.

Output Format:
Your final answer should be a single integer — the total number of valid partitions.
Example: `10` (do **NOT** include the backticks or quotes); this means there are 10 distinct ways to split {N} into {K} parts.
"""
    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the NumberPartitionCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 1, "N should be greater than or equal to 1"

        N = self.parameter["N"] = random.randint(1, MAX_N)
        K = self.parameter["K"] = random.randint(1, N)

        # Dynamic programming solution
        dpF = [[0 for _ in range(K + 1)] for _ in range(N + 1)]
        for i in range(1, N + 1) :
            dpF[i][1] = 1
            dpF[i][0] = 1
        for i in range(2, N + 1) :
            for x in range(2, K + 1) :
                if i > x :
                    dpF[i][x] = dpF[i - 1][x - 1] + dpF[i - x][x]
                else :
                    dpF[i][x] = dpF[i - 1][x - 1]
        self.parameter["reference_answer"] = dpF[N][K]
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], K = self.parameter["K"])


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