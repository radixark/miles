import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class GCDOne_Counting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2257
    prompt_template = \
r"""How many pairs (x, y) satisfy gcd(x, y) being exactly 1, where 1 ≤ x ≤ {N} and 1 ≤ y ≤ {M}? Here, gcd(x, y) denotes the **greatest common divisor** of integers x and y.

**Output Format:** Your final answer should be a single integer — the number of pairs (x, y) such that x and y are coprime, i.e., gcd(x, y) = 1."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the GCDOne_Counting_Environment instance.
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

        N = self.parameter["N"] = random.randint(2, MAX_N_M)
        M = self.parameter["M"] = random.randint(2, MAX_N_M)


        mu = [0] * (min(N, M) + 1)
        mu[1] = 1
        flag = [False] * (min(N, M) + 1)
        primes = []
        for i in range(2, min(N, M) + 1) :
            if not flag[i] :
                primes.append(i)
                mu[i] = -1
            for p in primes :
                ip = i * p
                if ip > min(N, M) :
                    break
                flag[ip] = True
                if i % p == 0 :
                    break
                else :
                    mu[ip] = -mu[i]

        f = [0] * (min(N, M) + 1)
        for i in range(1, min(N, M) + 1) :
            f[i] = mu[i]

        prefix = [0] * (min(N, M) + 1)
        s = 0
        for i in range(1, min(N, M) + 1) :
            s += f[i]
            prefix[i] = s

        ans = 0
        l = 1
        while l <= N and l <= M :
            an = N // l
            am = M // l
            r = min(N // an, M // am)
            ans += (prefix[r] - prefix[l-1]) * an * am
            l = r + 1
        
        self.parameter["reference_answer"] = ans
        assert ans > 0
    
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