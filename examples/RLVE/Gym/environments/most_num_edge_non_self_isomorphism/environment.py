import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MostNumEdge_NonSelfIsomorphism_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Consider a simple **undirected graph** G on {N} labeled vertices `1` to `{N}`. We say G is **asymmetric** if the only bijection (permutation) `p` of the vertices that preserves all edges (i.e., `(u, v)` is an edge iff `(p(u), p(v))` is an edge) is the identity permutation. What is the **maximum number of edges** an asymmetric graph G on {N} labeled vertices can have?"""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MostNumEdge_NonSelfIsomorphism_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 6, "MAX_N should be greater than or equal to 6"

        N = self.parameter["N"] = random.randint(6, MAX_N)


        def C(n, m) :
            if 0 > m or m > n :
                return 0
            ans = 1
            for i in range(m) :
                ans = ans * (n - i) // (i + 1)
            return ans
        f = h = [0 for i in range(0, N + 1)]
        g = [[0 for j in range(0, N + 1)] for i in range(0, N + 1)]
        g[0][0] = 1
        for i in range(1, N + 1):
            h[i] = g[i - 1][i - 1]
            for j in range(0, N + 1):
                for k in range(j // i + 1):
                    g[i][j] += C(h[i], k) * g[i - 1][j - i * k]
        for i in range(1, N + 1):
            f[i] = g[(i - 1) // 2][i - 1]
            if i % 2 == 0:
                f[i] += C(g[i // 2 - 1][i // 2 - 1], 2)

        res = N * (N - 1) // 2 - N
        original_N = N
        for i in range(1, original_N + 1):
            cnt = min(N // i, f[i])
            res += cnt
            N -= i * cnt
        self.parameter["reference_answer"] = res
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"])


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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]