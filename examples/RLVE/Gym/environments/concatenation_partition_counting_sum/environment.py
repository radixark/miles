import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class ConcatenationPartitionCountingSum_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3176
    prompt_template = \
r"""Define F[n] as follows:
- F[0] = 1  
- For all n ≥ 1: F[n] = sum(F[n - m] for m in range(1, min(n, {M}) + 1)) (Python-like syntax)

You are given a number string S: {S}
Consider all possible partitions of S into non-empty substrings s[1], s[2], ..., s[k] (for any k ≥ 1), such that concatenating s[1] through s[k] gives exactly {S}. Note that leading zeros are allowed in any s[i]. For each such partition, compute the value F[int(s[1]) + int(s[2]) + ... + int(s[k])]. Please compute the total sum of this value over all such partitions, modulo {MOD}."""
    
    def __init__(self,
                 max_MOD : int = 10000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the ConcatenationPartitionCountingSum_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_MOD = max_MOD
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

        S = self.parameter["S"] = "".join(random.choices("0123456789", k = N))

        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert M >= 1, "M should be greater than or equal to 1"

        MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)


        class Node:
            def __init__(self, init_zero=True):
                # initialize a MxM matrix of zeros
                self.a = [[0] * M for _ in range(M)] if init_zero else None

            def init(self):
                # companion matrix for transitions: P[0]
                for i in range(M):
                    self.a[i][M-1] = 1
                for i in range(1, M):
                    self.a[i][i-1] = 1

            def init1(self):
                # identity matrix
                for i in range(M):
                    self.a[i][i] = 1

            def __mul__(self, other):
                # matrix multiplication mod
                z = Node()
                for i in range(M):
                    for k in range(M):
                        if self.a[i][k] == 0:
                            continue
                        aik = self.a[i][k]
                        row_z = z.a[i]
                        row_o = other.a[k]
                        for j in range(M):
                            row_z[j] = (row_z[j] + aik * row_o[j]) % MOD
                return z

            def __add__(self, other):
                # matrix addition mod
                z = Node()
                for i in range(M):
                    for j in range(M):
                        z.a[i][j] = (self.a[i][j] + other.a[i][j]) % MOD
                return z


        def ksm(mat, exp):
            # fast exponentiation of matrix mat^exp
            res = Node()
            res.init1()
            base = mat
            e = exp
            while e > 0:
                if e & 1:
                    res = res * base
                base = base * base
                e >>= 1
            return res

        digits = [int(ch) for ch in S]

        # precompute P[i] = P^(10^i)
        P = [None] * N
        P[0] = Node()
        P[0].init()
        for i in range(1, N):
            P[i] = ksm(P[i-1], 10)

        # F[i][j]: transition matrix for substring S[i..j]
        F = [[None] * N for _ in range(N)]
        for j in range(N):
            for i in range(j, -1, -1):
                d = digits[i]
                if i == j:
                    F[i][j] = ksm(P[0], d)
                else:
                    # F[i][j] = F[i+1][j] * P[j-i]^d
                    t = ksm(P[j-i], d)
                    F[i][j] = F[i+1][j] * t

        # DP g: g[k] is matrix for prefix of length k
        g = [None] * (N + 1)
        # g[0] = identity
        g[0] = Node()
        g[0].init1()
        for i in range(1, N + 1):
            cur = Node()
            # sum over previous split points
            for j in range(i):
                cur = cur + (g[j] * F[j][i-1])
            g[i] = cur

        # answer: sum of first row of g[N]
        self.parameter["reference_answer"] = sum(g[N].a[0][i] for i in range(M)) % MOD
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(S = self.parameter["S"], M = self.parameter["M"], MOD = self.parameter["MOD"])


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