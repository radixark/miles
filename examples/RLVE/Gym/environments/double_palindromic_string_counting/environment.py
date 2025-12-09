import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class DoublePalindromicStringCounting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""We define a string `S` as **double palindromic** if it satisfies all of the following conditions:
- Each character in `S` is an integer between `1` and `{C}` (inclusive).
- `S` can be written as the concatenation of two **non-empty palindromic strings**, `S1` and `S2`, such that `S = S1 + S2`.

Please count the number of **distinct double palindromic strings** of length **at most** `{N}`.

**Output Format:** Your final answer should be a single integer — the total number of such distinct double palindromic strings.
"""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the DoublePalindromicStringCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 2, "MAX_N should be greater than or equal to 2"

        N = self.parameter["N"] = random.randint(2, MAX_N)

        assert "C" in self.parameter, "C is required in parameter"
        C = self.parameter["C"]
        assert C >= 1, "C should be greater than or equal to 1"


        def pre(N):
            mu      = [0] * (N+1)
            f_pref  = [0] * (N+1)
            is_comp = [False] * (N+1)
            primes  = []

            mu[1]     = 1
            f_pref[1] = 1

            for i in range(2, N+1):
                if not is_comp[i]:
                    primes.append(i)
                    mu[i]     = -1
                    f_pref[i] = 1 - i
                for p in primes:
                    ip = i * p
                    if ip > N:
                        break
                    is_comp[ip] = True
                    if i % p == 0:
                        f_pref[ip] = f_pref[i]
                        break
                    mu[ip]     = -mu[i]
                    f_pref[ip] = f_pref[i] * (1 - p)
            for i in range(1, N+1):
                mu[i]     += mu[i-1]
                f_pref[i] += f_pref[i-1]
            return mu, f_pref

        def S(n):
            return n*(n+1)//2

        def make_calc1(f_pref, N):
            memo = {}
            def calc1(n):
                if n <= N:
                    return f_pref[n]
                if n in memo:
                    return memo[n]
                res = n
                i   = 2
                while i <= n:
                    t    = n // i
                    last = n // t
                    res  -= (S(last) - S(i-1)) * calc1(t)
                    i    = last + 1
                memo[n] = res
                return res
            return calc1

        def make_calc2(mu_pref, N):
            memo = {}
            def calc2(n):
                if n <= N:
                    return mu_pref[n]
                if n in memo:
                    return memo[n]
                res = 1
                i   = 2
                while i <= n:
                    t    = n // i
                    last = n // t
                    res  -= (last - i + 1) * calc2(t)
                    i    = last + 1
                memo[n] = res
                return res
            return calc2

        def query1(n, C, den):
            # ((t*(4n-2) - 4*(t-C)/(C-1)) / (C-1))
            t = pow(C, n+1)
            # first subtract the geometric‐sum piece:
            part = 4 * (t - C) // den
            return (t * (4*n - 2) - part) // den

        def querysum(n, C, den):
            half = n // 2
            # sum up to half:
            s_half = query1(half, C, den)
            t      = pow(C, half+1)
            extra  = (n + half) if (n & 1) else half
            return s_half + t * extra

        def solve1(N, C, calc1_fn, den):
            ans = 0
            i   = 1
            while i <= N:
                t    = N // i
                last = N // t
                ans += (querysum(last, C, den) - querysum(i-1, C, den)) * calc1_fn(t)
                i    = last + 1
            return ans

        def query2(n, C, den):
            half = n // 2
            t    = pow(C, half+1)
            # 2*(t-C)/(C-1)  +  (t if odd)
            base = 2 * (t - C) // den
            return base + (t if (n & 1) else 0)

        def solve2(N, C, calc2_fn, den):
            ans = 0
            i   = 1
            while i <= N:
                t    = N // i
                last = N // t
                ans += (query2(last, C, den) - query2(i-1, C, den)) * calc2_fn(t)
                i    = last + 1
            return ans

        den   = C - 1                # we’ll just divide by this
        mu_pref, f_pref = pre(N)
        calc1_fn = make_calc1(f_pref, N)
        calc2_fn = make_calc2(mu_pref, N)
        answer = solve1(N, C, calc1_fn, den) - solve2(N, C, calc2_fn, den)
        
        self.parameter["reference_answer"] = answer
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], C = self.parameter["C"])


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