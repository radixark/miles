import random
from typing import Optional
from bisect import bisect_left
from Gym.environment import VerifiableEnvironment


class SegmentMinLengthEqual_Counting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/B3902
    prompt_template = \
r"""An array x[1], x[2], ..., x[{N}] is called **valid** if and only if there exists a partition of it into intervals such that the minimum value in each interval is exactly equal to the interval’s length. Equivalently, there exist indices 0 = x_1 < x_2 < ... < x_m = {N}, such that for every 1 ≤ i < m, we have min_{j = x_i + 1}^{x_{i+1}} a_j = x_{i+1} - x_i. What is the number of such valid arrays x, where each element x[i] must belong to the set S = {S}? Output the answer modulo {MOD}."""
    def __init__(self,
                 max_MOD : int = 1000000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the SegmentMinLengthEqual_Counting_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_MOD = max_MOD
        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        S = self.parameter["S"] = sorted(random.sample(range(1, N + 1), random.randint(2, N)))
        MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)


        def quick_power(a: int, e: int) -> int:
            # fast power mod MOD (you can also use pow(a, e, MOD) directly)
            res = 1
            a %= MOD
            while e:
                if e & 1:
                    res = (res * a) % MOD
                a = (a * a) % MOD
                e >>= 1
            return res

        def main(B):
            M = len(B)
            exist_set = set(B)

            # c[i] = count of elements in S >= i, for i in 1..N
            C = [0] * (N + 1)
            for i in range(1, N + 1):
                # number of elements >= i = M - index of first >= i
                C[i] = M - bisect_left(B, i)

            # DP
            F = [0] * (N + 1)
            F[0] = 1

            for i in range(1, N + 1):
                total = 0
                for j in range(i):
                    L = i - j  # length of the last segment
                    if L in exist_set:
                        cL = C[L]
                        # ways to fill a segment of length L with min exactly L:
                        ways = (quick_power(cL, L) - quick_power(cL - 1, L) + MOD) % MOD
                        total = (total + F[j] * ways) % MOD
                F[i] = total

            return F[N]
        
        self.parameter["reference_answer"] = main(S)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.replace(r"{N}", str(self.parameter["N"])).replace(r"{S}", "{" + ", ".join(map(str, self.parameter["S"])) + "}").replace(r"{MOD}", str(self.parameter["MOD"]))


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
            if not (0 <= processed_result < self.parameter["MOD"]) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]