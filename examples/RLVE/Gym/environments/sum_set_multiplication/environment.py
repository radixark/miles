import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SumSetMultiplication_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4463
    prompt_template = r"""Consider all sequences A[1..{N}] of **distinct integers** chosen from [1, {K}]. Compute the sum of (A[1] × A[2] × ... × A[{N}]) over all such sequences, modulo {MOD}."""
    MODs = (666623333, 998244353, 10 ** 9 + 7)

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the SumSetMultiplication_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 3, "MAX_N should be greater than or equal to 3"

        assert "MAX_K" in self.parameter, "MAX_K is required in parameter"
        MAX_K = self.parameter["MAX_K"]
        assert MAX_K > MAX_N, "MAX_K should be greater than MAX_N"

        N = self.parameter["N"] = random.randint(3, MAX_N)
        K = self.parameter["K"] = random.randint(N + 1, MAX_K)
        MOD = self.parameter["MOD"] = random.choice(self.MODs)


        # dynamic sizing based on N
        size = 2 * N + 3  # to safely index up to 2N+1 and use i+1 at i=2N
        F = [0] * size
        C = [0] * size

        def mod_pow(a, b):
            a %= MOD
            res = 1
            while b:
                if b & 1:
                    res = (res * a) % MOD
                a = (a * a) % MOD
                b >>= 1
            return res

        INX = K if (2 * N + 1) > K else (2 * N + 1)
        C[INX] = 1
        F[0] = 1

        for i in range(1, N + 1):
            for j in range(2 * i, 1, -1):
                F[j] = (F[j - 1] * j + F[j - 2] * (2 * i - j)) % MOD
            F[1] = F[0]
            F[0] = 0

        if INX == 2 * N + 1:
            for i in range(1, 2 * N + 1):
                C[INX] = (C[INX] * ((K - i) % MOD)) % MOD
                C[INX] = (C[INX] * mod_pow(i % MOD, MOD - 2)) % MOD

        for i in range(INX - 1, -1, -1):
            numerator = (K + 2 * N - i) % MOD
            denom = (K - i) % MOD
            C[i] = C[i + 1] * numerator % MOD * mod_pow(denom, MOD - 2) % MOD

        ans = 0
        for i in range(0, 2 * N + 1):
            ans = (ans + C[i] * F[i]) % MOD
        for i in range(1, N + 1):
            ans = ans * i % MOD
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], K = self.parameter["K"], MOD = self.parameter["MOD"])
    

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