import random
from typing import Optional
from Gym.environment import VerifiableEnvironment
import math


class DiscreteLogarithm_Environment(VerifiableEnvironment) : # Source : https://www.spoj.com/problems/MOD/
    prompt_template = \
r"""Please find the **smallest** non-negative integer **y** such that **({X}^y) MOD {Z} = {K} MOD {Z}**.

**Output Format:** Your final answer should be a single non-negative integer  — the smallest **y** satisfying the equation."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = 0.0, rewarding_strategy : str = "(gold/answer)^beta", rewarding_beta : float = 2.0, rewarding_weight : float = 1.0,
                 **kwargs) :
        """
        Initialize the DiscreteLogarithm_Environment instance.
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
        assert "MAX_Z" in self.parameter, "MAX_Z is required in parameter"
        assert self.parameter["MAX_Z"] >= 2, "MAX_Z should be greater than or equal to 2"
        Z = self.parameter["Z"] = random.randint(2, self.parameter["MAX_Z"])
        X = self.parameter["X"] = random.randint(2, Z)
        Y = self.parameter["Y"] = random.randint(2, Z)
        K = self.parameter["K"] = pow(X, Y, Z)

        def modular_log_solver(a, mod, r):

            def adjust(x, mod):
                return (x % mod + mod) % mod

            def check(x, mod):
                return adjust(x, mod)

            def power(a, n, mod):
                s = 1
                x = a % mod
                while n:
                    if n & 1:
                        s = s * x % mod
                    x = x * x % mod
                    n >>= 1
                return s

            def gcd(a, b):
                return math.gcd(a, b)

            def exgcd(a, b):
                if b == 0:
                    return (1, 0)
                else:
                    x1, y1 = exgcd(b, a % b)
                    x, y = y1, x1 - (a // b) * y1
                    return (x, y)

            def BSGS(a, r, mod):
                a %= mod
                r %= mod
                T = int(round(math.sqrt(mod)))
                a_T = power(a, T, mod)
                H = {}
                cur = r
                for i in range(1, T+1):
                    cur = cur * a % mod
                    H[cur] = i
                cur = a_T
                for i in range(1, T+2):
                    val = cur
                    if val in H:
                        return i * T - H[val]
                    cur = cur * a_T % mod
                return -1

            def exBSGS(a, r, mod):
                a %= mod
                r %= mod
                g = gcd(mod, a)
                if r % g != 0:
                    if r == 1:
                        return 0
                    else:
                        return -1
                if g == 1:
                    return BSGS(a, r, mod)
                else:
                    iv, y = exgcd(a // g, mod // g)
                    iv = check(iv, mod // g)
                    res = exBSGS(a, r // g * iv % (mod // g), mod // g)
                    if res < 0:
                        return -1
                    return res + 1

            x = exBSGS(a, r, mod)
            return x

        self.parameter["reference_answer"] = modular_log_solver(X, Z, K)
        assert self.parameter["reference_answer"] >= 0, "ans should be non-negative"
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(X = self.parameter["X"], Z = self.parameter["Z"], K = self.parameter["K"])


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

            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["rewarding_weight"]
            
            if pow(self.parameter["X"], processed_result, self.parameter["Z"]) != self.parameter["K"] :
                return self.rewards["invalid_answer"]

            assert processed_result >= self.parameter["reference_answer"], "processed_result should be greater than or equal to reference_answer"

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((self.parameter["reference_answer"] / processed_result) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                assert self.parameter["reference_answer"] != processed_result
                return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]