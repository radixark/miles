import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SumXorDivisorNum_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3791
    prompt_template = r"""Let d(n) denote the number of positive divisors of n (with d(0) = 0). What is the sum of d(i XOR j XOR {X}) (XOR means bitwise XOR) over all integer pairs (i, j) such that 0 ≤ i ≤ {N} and 0 ≤ j ≤ {M}?"""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SumXorDivisorNum_Environment instance.
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
        assert MAX_N_M >= 3, "MAX_N_M should be greater than or equal to 3"

        N = self.parameter["N"] = random.randint(3, MAX_N_M)
        M = self.parameter["M"] = random.randint(3, MAX_N_M)
        X = self.parameter["X"] = random.randint(0, MAX_N_M)

        
        A = N + 1
        B = M + 1

        # Build bit‐arrays (LSB first) of lengths exactly what we need
        a = []
        while A:
            a.append(A & 1)
            A >>= 1

        b = []
        while B:
            b.append(B & 1)
            B >>= 1

        x = []
        while X:
            x.append(X & 1)
            X >>= 1

        # Pad all to the same length
        L = max(len(a), len(b), len(x))
        a += [0] * (L - len(a))
        b += [0] * (L - len(b))
        x += [0] * (L - len(x))

        # h[i] = integer value of bits (a⊕b⊕x) from position i..L-1
        h = [0] * (L + 1)
        for i in range(L - 1, -1, -1):
            h[i] = h[i + 1] + ((a[i] ^ b[i] ^ x[i]) << i)

        # mi[k] = 2^k mod (we only need up to L-1)
        mi = [1] * L
        for i in range(1, L):
            mi[i] = mi[i - 1] * 2

        # Cache for the divisor‐summatory function
        sd = {}

        def D(val):
            """Return sum_{k=1}^val d(k) mod, where d(k)=#divisors of k.  d(0)=0."""
            if val <= 0:
                return 0
            if val in sd:
                return sd[val]
            res = 0
            l = 1
            # Standard sqrt‐decomposition trick to compute sum_{i=1..val} floor(val/i)
            while l <= val:
                t = val // l
                r = val // t
                cnt = r - l + 1
                res += cnt * t
                l = r + 1
            sd[val] = res
            return res

        # Main double loop over set bits in a[] and b[]
        ans = 0
        for i in range(L):
            if a[i] == 0:
                continue
            for j in range(L):
                if b[j] == 0:
                    continue
                s = max(i, j)
                t = min(i, j)

                # H = h[s] with the s-th bit of the XOR flipped once,
                # then flipped again if i==j (to undo double‐count)
                H = h[s] ^ (1 << s)
                if i == j:
                    H ^= (1 << s)

                # We want sum_{v=H .. H + 2^s - 1} d(v)
                val = D(H + (1 << s) - 1) - D(H - 1)
                ans += val * mi[t]

        assert ans > 0, "The answer should be greater than 0"
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], M = self.parameter["M"], X = self.parameter["X"])


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