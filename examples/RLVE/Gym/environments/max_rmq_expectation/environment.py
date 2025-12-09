import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MaxRMQExpectation_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3600
    prompt_template = \
r"""Let's randomly generate an array A[1], ..., A[{N}], where each A[i] is independently and uniformly chosen from the integers 1 to {X} (so there are {X}^{N} possible arrays in total). You are also given {Q} intervals [L[i], R[i]] (1 ≤ i ≤ {Q}):
{intervals}

For each interval [L[i], R[i]], define M[i] = min(A[j]) for L[i] ≤ j ≤ R[i]. Please compute the **expected value** of max(M[1], ..., M[{Q}]) and output the result **modulo {MOD}**."""
    MODs = (666623333, 998244353, 10 ** 9 + 7)
    
    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MaxRMQExpectation_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        X = self.parameter["X"] = random.randint(2, N)
        Q = self.parameter["Q"] = random.randint(1, N)

        self.parameter["intervals"] = intervals = []
        for _ in range(Q) :
            L, R = random.randint(1, N), random.randint(1, N)
            if L > R :
                L, R = R, L
            intervals.append((L, R))


        MOD = self.parameter["MOD"] = random.choice(self.MODs)

        def modinv(a):
            # modular inverse via Fermat's little theorem
            return pow(a, MOD - 2, MOD)

        def compute():
            # ar[i] will store the maximum l among all queries whose r+1 == i
            ar = [0] * (N + 2)
            for l, r in intervals:
                ar[r + 1] = max(ar[r + 1], l)
            # take prefix max so that ar[j] = max_{i ≤ j}( ar[i] )
            for i in range(1, N + 2):
                if ar[i] < ar[i - 1]:
                    ar[i] = ar[i - 1]

            # ix = 1/X mod
            ix = modinv(X)
            ans = 0

            # loop over possible threshold i = 1..X
            for i1 in range(1, X + 1):
                # p = (i1 - 1) / X  (mod)
                p = (i1 - 1) * ix % MOD
                one_minus_p = (1 - p) % MOD
                # ip = (1 - p)^{-1} mod
                ip = modinv(one_minus_p)

                # precompute ff0[j] = (1-p)^j, ff1[j] = ip^j
                ff0 = [1] * (N + 1)
                ff1 = [1] * (N + 1)
                for j in range(1, N + 1):
                    ff0[j] = ff0[j - 1] * one_minus_p % MOD
                    ff1[j] = ff1[j - 1] * ip        % MOD

                # f0[j], f1[j] DP arrays
                f0 = [0] * (N + 1)
                f1 = [0] * (N + 1)
                f1[0] = 1
                for j in range(1, N + 1):
                    if ar[j] > 0:
                        prev = (f1[j - 1] - f1[ar[j] - 1]) % MOD
                    else:
                        prev = f1[j - 1]
                    # f0[j] = p * prev * (1-p)^(j-1)
                    f0[j] = p * prev % MOD * ff0[j - 1] % MOD
                    # f1[j] = f1[j-1] + f0[j]*(ip^j)
                    f1[j] = (f1[j - 1] + f0[j] * ff1[j]) % MOD

                # sum up contributions from j = ar[N+1]..N
                Lmax = ar[N + 1]
                s = 0
                for j in range(Lmax, N + 1):
                    s = (s + f0[j] * ff0[N - j]) % MOD

                # accumulate into answer: ans += 1 - s
                ans = (ans + 1 - s) % MOD

            return ans
        self.parameter["reference_answer"] = compute()
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            X = self.parameter["X"],
            Q = self.parameter["Q"],
            intervals = "\n".join("[{}, {}]".format(L, R) for L, R in self.parameter["intervals"]),
            MOD = self.parameter["MOD"],
        )
    

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