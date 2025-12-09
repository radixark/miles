import random
from functools import cmp_to_key
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class ProtectingFlowers_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2878
    prompt_template = \
r"""You are given two arrays `T` and `D`, each containing {N} integers:
{T_and_D}

Please output **a permutation of 1 to {N}**, denoted as p[1], p[2], ..., p[{N}] ({N} integers in one line) with adjacent numbers separated by spaces:
- Define S[i] as the sum of T[p[j]] for all 1 ≤ j < i (so S[1] = 0).
- The objective is to minimize the total sum of S[i] * D[p[i]] for i from 1 to {N}."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the ProtectingFlowers_Environment instance
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
        assert N >= 3, "N should be greater than or equal to 3"

        T = self.parameter["T"] = [random.randint(1, N) for _ in range(N)]
        D = self.parameter["D"] = [random.randint(1, N) for _ in range(N)]


        A = []
        for t, d in zip(T, D):
            A.append((t, d))

        def cmp(x, y):
            # sort by t/d ascending without floating point
            left = x[0] * y[1]
            right = x[1] * y[0]
            if left < right:
                return -1
            elif left > right:
                return 1
            else:
                return 0
        A.sort(key=cmp_to_key(cmp))

        # prefix sums of d
        prefix = [0] * (N + 1)
        for i in range(N):
            prefix[i + 1] = prefix[i] + A[i][1]

        ans = 0
        total_d = prefix[N]
        for i in range(N):
            t_i, d_i = A[i]
            # cows after i (in sorted order) keep eating while we fetch i
            ans += t_i * (total_d - prefix[i + 1])

        assert ans > 0, "The answer should be positive"
        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            T_and_D = "\n".join("T[{}]={} D[{}]={}".format(i, Ti, i, Di) for i, (Ti, Di) in enumerate(zip(self.parameter["T"], self.parameter["D"]), start = 1)),
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
            if set(processed_result) != set(range(1, self.parameter["N"] + 1)) :
                return self.rewards["invalid_solution"]

            answer, gold = 0, self.parameter["gold_answer"]
            T, D = [None] + self.parameter["T"], [None] + self.parameter["D"]
            S = [0] * (self.parameter["N"] + 1)
            for i, Pi in enumerate(processed_result, start = 1) :
                S[i] = S[i - 1] + T[Pi]
                answer += S[i - 1] * D[Pi]

            assert 0 < gold <= answer, "gold should be less than or equal to answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]