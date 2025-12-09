import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SumGCD_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4449
    prompt_template = \
r"""Please compute sum(GCD(i, j)^{K}) for all pairs (i, j) such that 1 ≤ i ≤ {N} and 1 ≤ j ≤ {M}. Here, GCD(i, j) denotes the **greatest common divisor** of integers i and j, and x^{K} denotes x raised to the power of K.

**Output Format:** Your final answer should be a single integer — the sum of GCD(i, j)^{K} over all such pairs."""

    def __init__(self,
                 max_K : int = 5,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SumGCD_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_K = max_K
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
        K = self.parameter["K"] = random.randint(1, self.max_K)


        is_comp = [False] * (min(N, M) + 1)
        f      = [0]     * (min(N, M) + 1)
        primes = []
        g      = []

        f[1] = 1
        for i in range(2, min(N, M) + 1) :
            if not is_comp[i] :
                primes.append(i)
                gi = i ** K
                g.append(gi)
                f[i] = (gi - 1)

            for j, p_j in enumerate(primes) :
                ip = i * p_j
                if ip > min(N, M) :
                    break
                is_comp[ip] = True
                if i % p_j == 0 :
                    f[ip] = f[i] * g[j]
                    break
                else :
                    f[ip] = f[i] * f[p_j]

        for i in range(1, min(N, M) + 1) :
            f[i] = (f[i] + f[i - 1])
        
        ans = 0
        i = 1
        while i <= min(N, M) :
            ni = N // i
            mi = M // i
            nxt = min(N // ni, M // mi)
            s = (f[nxt] - f[i - 1])
            ans += s * ni * mi
            i = nxt + 1
        
        self.parameter["reference_answer"] = ans
        assert ans > 0, "ans should be greater than 0"
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], M = self.parameter["M"], K = self.parameter["K"])


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