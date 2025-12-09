import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class IntegerFactorizationCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3861
    prompt_template = \
r"""Count the number of ways to factorize {N} into (multiple, i.e., more than 1) **distinct** positive integers greater than 1 such that their product is {N}. The order of factors does not matter. For example, $688 = 2 × 4 × 86 = 2 × 8 × 43 = 2 × 344 = 4 × 172 = 8 × 86 = 16 × 43$, so there are 6 valid ways in total."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the IntegerFactorizationCountingProblem instance.
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
        assert MAX_N >= 4, "MAX_N should be greater than or equal to 4"

        N = self.parameter["N"] = random.randint(4, MAX_N)

        
        def count_factorizations(N: int) -> int:
            # 1. enumerate divisors of N
            divs = []
            i = 1
            while i * i <= N:
                if N % i == 0:
                    divs.append(i)
                    if i != N // i:
                        divs.append(N // i)
                i += 1
            divs.sort()
            total = len(divs)

            # 2. map each divisor to its index (0-based)
            idx = {d: i for i, d in enumerate(divs)}

            # 3. dp[i] = number of ways to get product = divs[i] using distinct divisors seen so far
            dp = [0] * total
            dp[0] = 1  # one way to make 1 (the empty product)

            # 4. for each divisor x = divs[j] (skip the first which is 1),
            #    update dp in place from high i down to j
            for j in range(1, total):
                xj = divs[j]
                for i in range(total - 1, j - 1, -1):
                    di = divs[i]
                    if di % xj == 0:
                        dp[i] += dp[idx[di // xj]]

            # 5. dp[total-1] counts also the trivial factorization [N] → subtract 1
            return dp[total - 1] - 1
        self.parameter["reference_answer"] = count_factorizations(N)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"])
    

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
            if processed_result < 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                if self.parameter["reference_answer"] == 0 :
                    return self.rewards["rewarding_weight"] * (processed_result == 0)
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]