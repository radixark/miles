import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class CycleCounting_Environment(VerifiableEnvironment) : # Source : https://codeforces.com/problemset/problem/11/D
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The graph contains the following undirected edges:
{edges}

Please count the number of simple cycles in the graph. A simple cycle is a cycle with at least 3 vertices, with no repeated vertices or edges.
Two cycles are considered equivalent if they consist of the same set of edges, regardless of the order or starting point; for example, the cycles `(0, 1, 2, 3)` and `(1, 0, 3, 2)` are identical, while `(0, 1, 2, 3)` and `(1, 0, 2, 3)` are NOT.

Output Format: Your final answer should be a single line containing the number of simple cycles in the graph.
"""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the CycleCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        edges = self.parameter["edges"] = random.sample([(u, v) for u in range(N) for v in range(u + 1, N)], int(edge_density * N * (N - 1) / 2))
        random.shuffle(edges)
        
        assert len(edges) == len(set(edges)), "edges should be unique"
        adjacent = [[False] * N for u in range(N)]
        for u, v in edges :
            assert 0 <= u < v < N
            adjacent[u][v] = adjacent[v][u] = True
        

        dpF = [[0] * N for S in range(1 << N)]
        for end in range(N) :
            dpF[1 << end][end] = 1
        answer = 0
        for S in range(1, 1 << N) :
            lowindex = 0
            while (1 << lowindex) != (S & -S) :
                lowindex += 1
            nowS = S
            while nowS :
                end = 0
                while (1 << end) != (nowS & -nowS) :
                    end += 1
                nowS ^= (1 << end)
                if not dpF[S][end] :
                    continue
                if adjacent[end][lowindex] :
                    if S - (1 << lowindex) - (1 << end) > 0 :
                        answer += dpF[S][end]
                nowR = ((1 << N) - 1) - S
                while nowR :
                    next = 0
                    while (1 << next) != (nowR & -nowR) :
                        next += 1
                    nowR ^= (1 << next)
                    if S & (1 << next) :
                        assert False, "next should not be in S"
                    if next < lowindex :
                        continue
                    if not adjacent[end][next] :
                        continue
                    dpF[S | (1 << next)][next] += dpF[S][end]
        self.parameter["reference_answer"] = answer // 2
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
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
            if processed_result < 0 :
                return self.rewards["wrong_format"]
            
            if self.parameter["reference_answer"] == 0 :
                return self.rewards["rewarding_weight"] * (processed_result == 0)

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]