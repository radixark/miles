import heapq
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class HamiltonianPathExistence_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a **directed graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The graph contains the following directed edges. Each edge is represented as a tuple `(s, t)`, meaning there is a directed edge **from vertex `s` to vertex `t`**:
{edges}

Please find a path `p_1, p_2, ..., p_{N}` such that the path **visits every vertex exactly once** (revisiting vertices is NOT allowed).

Output Format:
Your final answer should be a single line containing the path in order: `p_1, p_2, ..., p_{N}`, separated by **spaces**.
Example: `0 2 1` (do **NOT** include the backticks or quotes); this means the path starts at vertex 0, then goes to vertex 2, and finally to vertex 1 (assuming 3 vertices in total)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(existing/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the HamiltonianPathExistence_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        edges = self.parameter["edges"] = []

        constructed_path = list(range(N))
        random.shuffle(constructed_path)
        self.parameter["reference_answer"] = " ".join(map(str, constructed_path))
        for s, t in zip(constructed_path, constructed_path[1 :]) :
            edges.append((s, t))
        
        num_edges = int(edge_density * N * (N - 1))
        if len(edges) < num_edges :
            remaining_edges = list(set((s, t) for s in range(N) for t in range(N) if s != t) - set(edges))
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            edges += remaining_edges
        random.shuffle(edges)
        
        assert len(edges) == len(set(edges)), "edges should be unique"
        for s, t in edges :
            assert 0 <= s < N, "s should be in range"
            assert 0 <= t < N, "t should be in range"
            assert s != t, "s should not be equal to t"
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(s, t) for s, t in self.parameter["edges"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            path = processed_result
            if len(path) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if set(path) != set(range(self.parameter["N"])) :
                return self.rewards["invalid_solution"]
            
            edges = set(map(tuple, self.parameter["edges"]))
            existing = sum(int((s, t) in edges) for s, t in zip(path, path[1 :]))
            assert existing <= self.parameter["N"] - 1, "existing should be less than or equal to len(path) - 1"
            
            if self.rewards["rewarding_strategy"] == "(existing/all)^beta" :
                return self.rewards["rewarding_weight"] * ((existing / (self.parameter["N"] - 1)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "existing=all" :
                return self.rewards["rewarding_weight"] * (existing == (self.parameter["N"] - 1))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]