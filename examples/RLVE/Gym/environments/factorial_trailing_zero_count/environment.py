import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class FactorialTrailingZeroCount_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3927
    prompt_template = r"""Compute {N}! (the factorial of {N}; {N} is in base 10) and express the result in base {K}. What's the number of trailing zeros in this base-{K} representation?"""
    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the FactorialTrailingZeroCount_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_K" in self.parameter, "MAX_N_K is required in parameter"
        MAX_N_K = self.parameter["MAX_N_K"]
        assert MAX_N_K >= 10, "MAX_N_K should be greater than or equal to 10"

        N, K = self.parameter["N"], self.parameter["K"] = random.randint(3, MAX_N_K), random.randint(2, MAX_N_K)


        # Factorize K into primes: K = prod p_i^{c_i}
        P = []
        C = []
        i = 2
        while i * i <= K:
            if K % i == 0:
                cnt = 0
                while K % i == 0:
                    K //= i
                    cnt += 1
                P.append(i)
                C.append(cnt)
            i += 1
        if K > 1:
            P.append(K)
            C.append(1)

        # Compute the limiting factor: min_i floor(v_p_i(N!) / c_i)
        ans = None
        for idx in range(len(P)):
            p = P[idx]
            exp = 0
            now = N
            while now:
                now //= p
                exp += now
            t = exp // C[idx]
            if ans is None or t < ans:
                ans = t

        self.parameter["reference_answer"] = ans if ans is not None else 0
    

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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]