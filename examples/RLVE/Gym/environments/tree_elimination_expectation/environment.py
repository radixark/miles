import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class TreeElimination_Expectation_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a **tree** with {N} vertices labeled from `1` to `{N}`, where vertex `1` is the **root** of the tree. Each vertex (except the root `1`) has a parent, specified as follows:
{parents}

Initially, **all vertices are uncolored**. In each step, you randomly select an **uncolored vertex** (with equal probability) and color all vertices on the entire path from the selected vertex to the root.

Please compute the **expected number of steps** required until **all vertices are colored**. Please give the expectation **modulo 10^9 + 7**.

**Output Format:** Your final answer should be a single integer — the expected number of steps modulo 10^9 + 7."""
    MOD = 10 ** 9 + 7

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the TreeElimination_Expectation_Environment instance.
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

        P = list(range(2, N + 1))
        random.shuffle(P)
        P = [1] + P

        parents = self.parameter["parents"] = []
        for i in range(1, N) :
            parent, u = P[random.randint(0, i - 1)], P[i]
            parents.append((parent, u))
        

        def mod_inverse(a : int) -> int :
            return pow(a, self.MOD - 2, self.MOD)

        def dfs(u : int, children : list[list[int]], size : list[int], fac : list[int], inv : list[int]) -> int :
            total = 0
            size[u] = 1
            for v in children[u] :
                total += dfs(v, children, size, fac, inv)
                size[u] += size[v]
            total += fac[size[u] - 1] * inv[size[u]] % self.MOD
            return total % self.MOD

        children : list[list[int]] = [[] for _ in range(N + 1)]
        for parent, u in parents :
            children[parent].append(u)

        fac = [1] * (N + 1)
        for i in range(1, N + 1) :
            fac[i] = fac[i - 1] * i % self.MOD
        inv = [1] * (N + 1)
        inv[N] = mod_inverse(fac[N])
        for i in range(N, 0, -1) :
            inv[i - 1] = inv[i] * i % self.MOD

        size = [0] * (N + 1)
        self.parameter["reference_answer"] = dfs(1, children, size, fac, inv)
        assert size[1] == N, "size[1] should be equal to N"
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            parents = "\n".join("parent[{}]={}".format(u, parent) for parent, u in self.parameter["parents"]),
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
            if not (0 <= processed_result < self.MOD) :
                return self.rewards["wrong_range"]
            
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]