import random
from math import gcd
from typing import Optional
from Gym.environment import VerifiableEnvironment


class DistinctEdgeColoredCompleteGraphCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4128
    prompt_template = r"""Consider all **complete undirected graphs** on vertices 1, 2, ..., {N}, where each edge is assigned a color from {M} colors (labeled from 1 to {M}). Two such graphs G and G' are considered **the same** if there exists a permutation p of the vertices such that for every unordered pair (u, v), the color of edge (u, v) in G equals the color of edge (p(u), p(v)) in G'. What's the number of **distinct** graphs under this equivalence (i.e., the number of non-isomorphic M-colored complete graphs on N vertices) (output the result modulo {MOD})?"""
    MODs = (666623333, 998244353, 10 ** 9 + 7)
    def __init__(self,
                wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                **kwargs) :
        """
        Initialize the DistinctEdgeColoredCompleteGraphCountingProblem instance.
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
        assert N >= 3, "N should be greater than or equal to 3"

        M = self.parameter["M"] = random.randint(2, N * (N - 1) // 2)
        MOD = self.parameter["MOD"] = random.choice(self.MODs)


        # Modular exponentiation
        def qPow(b, e):
            a = 1
            b %= MOD
            while e:
                if e & 1:
                    a = (a * b) % MOD
                b = (b * b) % MOD
                e >>= 1
            return a

        # Precompute inverses, factorials, inverse factorials up to N
        Inv = [0] * (N + 1)
        Fac = [0] * (N + 1)
        iFac = [0] * (N + 1)

        def Init(limit):
            Inv[1] = 1
            for i in range(2, limit + 1):
                Inv[i] = (MOD - MOD // i) * Inv[MOD % i] % MOD
            Fac[0] = 1
            iFac[0] = 1
            for i in range(1, limit + 1):
                Fac[i] = (Fac[i - 1] * i) % MOD
                iFac[i] = (iFac[i - 1] * Inv[i]) % MOD

        # Globals mirroring the C++ code
        Sum = 0
        stk = [0]  # sentinel to mimic C++ global zero-initialized array
        t = 0
        n1 = 0
        n2 = 1

        def DFS(s, mx, c):
            nonlocal Sum, t, n1, n2
            if s == 0:
                Sum = (Sum + qPow(M, n1) * n2) % MOD
                return
            a = n1
            b = n2
            for i in range(1, mx + 1):
                stk.append(i)
                t += 1
                n1 = a + i // 2
                for j in range(1, t):
                    n1 += gcd(stk[j], i)
                n2 = b * Inv[i] % MOD
                if i == stk[t - 1]:
                    n2 = n2 * Fac[c] % MOD * iFac[c + 1] % MOD
                DFS(s - i, min(s - i, i), c + 1 if i == stk[t - 1] else 1)
                t -= 1
                stk.pop()

        # Run
        Init(N)
        DFS(N, N, 0)
        self.parameter["reference_answer"] = Sum


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