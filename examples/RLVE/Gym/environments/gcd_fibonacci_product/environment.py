import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class GCDFibonacciProduct_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3704
    prompt_template = \
r"""The Fibonacci sequence is defined as follows: f(0) = 0, f(1) = 1, and f(n) = f(n - 1) + f(n - 2) for all n ≥ 2.
Please compute the product of all f(gcd(i, j)) for all pairs (i, j) such that 1 ≤ i ≤ {N} and 1 ≤ j ≤ {M}. Output the result modulo {MOD}."""
    MODs = (666623333, 998244353, 10 ** 9 + 7)
    
    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the GCDFibonacciProduct_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 3, "MAX_N_M should be greater than or equal to 3"

        N = self.parameter["N"] = random.randint(3, MAX_N_M)
        M = self.parameter["M"] = random.randint(3, MAX_N_M)

        MOD = self.parameter["MOD"] = random.choice(self.MODs)


        def init(max_n):
            # Linear sieve to compute mu[1..max_n]
            is_composite = [False] * (max_n + 1)
            primes = []
            mu = [0] * (max_n + 1)
            mu[1] = 1
            for i in range(2, max_n + 1):
                if not is_composite[i]:
                    primes.append(i)
                    mu[i] = -1
                for p in primes:
                    if i * p > max_n:
                        break
                    is_composite[i * p] = True
                    if i % p == 0:
                        mu[i * p] = 0
                        break
                    else:
                        mu[i * p] = -mu[i]

            # f and fr arrays
            f = [1] * (max_n + 1)
            fr = [1] * (max_n + 1)

            A, B = 1, 0
            for i in range(1, max_n + 1):
                # update the alternating Fibonacci-like sequence
                B = (A + B) % MOD
                A = (B - A) % MOD
                # precompute factors
                invB = pow(B, MOD - 2, MOD)     # modular inverse of B
                for j in range(i, max_n + 1, i):
                    k = j // i
                    m = mu[k]
                    # apply to f[j]
                    if m == -1:
                        f[j] = f[j] * invB % MOD
                    elif m == 0:
                        # multiply by 1 — no change
                        pass
                    else:  # m == 1
                        f[j] = f[j] * B % MOD
                    # apply to fr[j]
                    # note: fr uses G[1 - mu[k]]
                    if m == 1:
                        fr[j] = fr[j] * invB % MOD
                    elif m == 0:
                        pass
                    else:  # m == -1
                        fr[j] = fr[j] * B % MOD

            # take prefix products
            for i in range(1, max_n + 1):
                f[i] = f[i-1] * f[i] % MOD
                fr[i] = fr[i-1] * fr[i] % MOD

            return f, fr

        f, fr = init(max(N, M))

        if N > M:
            N, M = M, N
        ans = 1
        i = 1
        while i <= N:
            divN = N // i
            divM = M // i
            j = min(N // divN, M // divM)
            base = f[j] * fr[i-1] % MOD
            exponent = divN * divM
            ans = ans * pow(base, exponent, MOD) % MOD
            i = j + 1
        self.parameter["reference_answer"] = ans


    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], M = self.parameter["M"], MOD = self.parameter["MOD"])


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