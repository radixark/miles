import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class CheckAllCycleXorZero_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3907
    prompt_template = \
r"""We have an **undirected graph** with {N} vertices labeled from `1` to `{N}`. The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning an undirected edge connects vertex `u` to vertex `v` with weight `w`:
{edges}

A cycle is defined as a path that starts and ends at the same vertex. Determine whether **every** cycle in the graph has an XOR sum of its edge weights equal to 0; output YES if the condition holds for every cycle in the graph, otherwise output NO."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the CheckAllCycleXorZero_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        weight_range = 2 ** (N * (N - 1) // 2).bit_length() - 1

        edges = self.parameter["edges"] = []

        permutations = list(range(1, N + 1))
        random.shuffle(permutations)
        XORs = [0] * (N + 1)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v, w = vertex, random.choice(permutations[: index]), random.randint(0, weight_range)
            XORs[u] = XORs[v] ^ w
            u, v = min(u, v), max(u, v)
            edges.append((u, v, w))
        
        must_YES = random.choice(["YES", "NO"])
        self.parameter["reference_answer"] = "YES"

        assert "edge_ratio" in self.parameter, "edge_ratio is required in parameter"
        edge_ratio = self.parameter["edge_ratio"]
        num_edges = int(edge_ratio * N)
        if len(edges) < num_edges :
            remaining_edges = list(set((u, v) for u in range(1, N + 1) for v in range(u + 1, N + 1)) - set((u, v) for u, v, w in edges))
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for u, v in remaining_edges :
                if must_YES == "YES" :
                    w = XORs[u] ^ XORs[v]
                else :
                    w = random.randint(0, weight_range)
                if (XORs[u] ^ XORs[v]) != w :
                    self.parameter["reference_answer"] = "NO"
                edges.append((u, v, w))
        else :
            assert False, "The number of edges should be less than num_edges"
        if must_YES == "YES" :
            assert self.parameter["reference_answer"] == "YES", "The reference answer should be YES"
        random.shuffle(edges)

        for u, v, w in edges :
            assert 1 <= u < v <= N
            assert 0 <= w <= weight_range, "edge weight should be within the specified range"
        assert len(edges) == len(set((u, v) for u, v, w in edges)), "edges should be unique"
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            edges = "\n".join("({}, {}, {})".format(u, v, w) for u, v, w in self.parameter["edges"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            return answer.strip()
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result not in ("YES", "NO") :
                return self.rewards["invalid_answer"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]