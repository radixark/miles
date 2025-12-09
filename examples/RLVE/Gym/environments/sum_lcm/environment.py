import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SumLCM_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1829
    prompt_template = \
r"""Please compute sum(LCM(i, j)) for all pairs (i, j) such that 1 ≤ i ≤ {N} and 1 ≤ j ≤ {M}. Here, LCM(i, j) denotes the **least common multiple** of integers i and j.

**Output Format:** Your final answer should be a single integer — the sum of LCM(i, j) over all such pairs."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SumLCM_Environment instance.
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


        max_rep = max(N, M)

        mu   = [0] * (max_rep + 1)
        pref = [0] * (max_rep + 1)
        mu[1] = 1
        primes = []
        vis = bytearray(max_rep + 1)

        for i in range(2, max_rep + 1) :
            if not vis[i] :
                primes.append(i)
                mu[i] = -1
            for p in primes :
                ip = i * p
                if ip > max_rep :
                    break
                vis[ip] = 1
                if i % p == 0 :
                    mu[ip] = 0
                    break
                mu[ip] = -mu[i]

        for i in range(1, max_rep + 1) :
            pref[i] = pref[i - 1] + mu[i] * i * i

        def tri(t : int) -> int :
            return (1 + t) * t // 2

        ans = 0
        for d in range(1, max_rep + 1) :
            nx, ny   = N // d, M // d
            limit    = nx if nx < ny else ny
            l = 1
            subtotal = 0
            while l <= limit :
                r  = min(nx // (nx // l), ny // (ny // l))
                mu_segment = pref[r] - pref[l - 1]
                sx = tri(nx // l)
                sy = tri(ny // l)
                subtotal = subtotal + mu_segment * sx * sy
                l = r + 1
            ans = ans + subtotal * d

        self.parameter["reference_answer"] = ans
    
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