import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SelfPowerSequenceMOD_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4139
    prompt_template = r"""Define $a[0] = 1$, and $a[n] = 2^(a[n-1])$. Let $b[n] = a[n] \bmod {MOD}$. It can be proven that $b[n]$ becomes constant after some point. Find this constant value."""

    
    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the SelfPowerSequenceMOD_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_MOD" in self.parameter, "MAX_MOD is required in parameter"
        MAX_MOD = self.parameter["MAX_MOD"]
        assert MAX_MOD >= 3, "MAX_MOD should be greater than or equal to 3"

        self.parameter["MOD"] = MOD = random.randint(3, MAX_MOD)


        def phi(n):
            ret = n
            i = 2
            while i * i <= n:
                if n % i == 0:
                    while n % i == 0:
                        n //= i
                    ret = ret // i * (i - 1)
                i += 1
            if n > 1:
                ret = ret // n * (n - 1)
            return ret

        def pow_mod(x, p, mod):
            ret = 1
            x %= mod
            while p:
                if p & 1:
                    ret = ret * x % mod
                x = x * x % mod
                p >>= 1
            return ret

        def solve(p):
            if p == 1:
                return 0
            t = phi(p)
            return pow_mod(2, solve(t) + t, p)

        self.parameter["reference_answer"] = solve(MOD)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(MOD = self.parameter["MOD"])
    

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