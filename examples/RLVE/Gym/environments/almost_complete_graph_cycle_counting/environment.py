import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class AlmostCompleteGraphCycleCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3862
    prompt_template = \
r"""Consider a graph with {N} vertices labeled from 1 to {N}. Every pair of vertices is connected by an undirected edge, except for the edge between vertices 1 and {N} (so the graph has {N} × ({N} - 1) / 2 - 1 edges).

What's the number of **simple cycles** in this graph? A simple cycle must:
- Have at least 3 vertices,
- Contain no repeated vertices or edges,
- Be considered the same as any cycle with the same set of edges (regardless of order or starting point); for example, `(1, 2, 3, 4)` and `(2, 1, 4, 3)` are the same, but `(1, 2, 3, 4)` and `(2, 1, 3, 4)` are different.
Output the answer modulo {MOD}."""

    def __init__(self,
                 max_MOD : int = 1000000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the AlmostCompleteGraphCycleCounting_Environment instance.
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
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 4, "MAX_N should be greater than or equal to 4"

        N = self.parameter["N"] = random.randint(4, MAX_N)

        MOD = self.parameter["MOD"] = 2 * random.randint(1, self.max_MOD // 2) + 1


        INV2 = (MOD + 1) // 2

        def calc(x, y, s, N):
            """
            x: current count of cycles for K_s
            y: current count of paths of length 1 (one edge) in K_s
            s: starting i value (we've precomputed up to K_s)
            N: target N
            """
            for i in range(s, N):
                # compute ((i-1)*(i-2)/2) % MOD efficiently
                half = ((i - 1) % MOD) * ((i - 2) % MOD) % MOD * INV2 % MOD
                x = (x + y * half) % MOD
                y = (y * ((i - 2) % MOD) + 1) % MOD
            # finally add the contribution for closing the cycle at N
            half_n = ((N - 2) % MOD) * ((N - 3) % MOD) % MOD * INV2 % MOD
            return (x + y * half_n) % MOD

        if N <= 3 :
            self.parameter["reference_answer"] = 0
        else :
            self.parameter["reference_answer"] = calc(1, 2, 4, N)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], MOD = self.parameter["MOD"])
    

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