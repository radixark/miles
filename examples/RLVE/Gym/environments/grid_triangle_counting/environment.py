import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class GridTriangleCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3166
    prompt_template = r"""How many non-degenerate triangles have all three vertices located at integer coordinate points (x, y) where 0 ≤ x ≤ {N} and 0 ≤ y ≤ {M}?"""


    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the GridTriangleCounting_Environment instance.
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
        assert MAX_N_M >= 1, "MAX_N_M must be at least 1"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(1, MAX_N_M), random.randint(1, MAX_N_M)


        if N > M:
            N, M = M, N

        # Sieve to compute phi up to N
        phi = [0] * (N + 1)
        mark = [False] * (N + 1)
        primes = []
        phi[1] = 1
        for i in range(2, N + 1):
            if not mark[i]:
                primes.append(i)
                phi[i] = i - 1
            for p in primes:
                ip = i * p
                if ip > N:
                    break
                mark[ip] = True
                if i % p == 0:
                    phi[ip] = phi[i] * p
                    break
                else:
                    phi[ip] = phi[i] * (p - 1)

        # Combination function C(x, 3) = x*(x-1)*(x-2)/6
        def C(x):
            return x * (x - 1) * (x - 2) // 6

        # Compute the contribution from degenerate (colinear) triples
        degenerate = 0
        for d in range(2, N + 1):
            term = phi[d]
            term *= (N - d + N % d + 2) * (N // d)
            term *= (M - d + M % d + 2) * (M // d)
            degenerate += term // 2

        # Total number of triples of points minus colinear ones
        total_points = (N + 1) * (M + 1)
        total_triples = C(total_points)
        subtract_N_lines = (M + 1) * C(N + 1)
        subtract_M_lines = (N + 1) * C(M + 1)

        self.parameter["reference_answer"] = total_triples - subtract_N_lines - subtract_M_lines - degenerate
        assert self.parameter["reference_answer"] > 0
    

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