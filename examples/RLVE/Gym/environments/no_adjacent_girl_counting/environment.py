import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class NoAdjacentGirlCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3223
    prompt_template = r"""Please count the number of ways to arrange {N} distinct boys, {M} distinct girls, and 2 distinct teachers in a line such that no two girls are adjacent and the two teachers are not adjacent."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the PalindromePartitionCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        while True :
            N, M = self.parameter["N"], self.parameter["M"] = random.randint(1, MAX_N_M), random.randint(1, MAX_N_M)
            Ans = 0
            def A(x, y) :
                res = 1
                for i in range(y) :
                    res *= x - i
                return res
            if N + 3 >= M :
                Ans += A(N + 3, M) * A(N + 2, N + 2)
            if N + 2 >= M :
                Ans -= 2 * A(N + 2, M) * A(N + 1, N + 1)
            if Ans > 0 :
                self.parameter["reference_answer"] = Ans
                break
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], M = self.parameter["M"])
    

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
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]