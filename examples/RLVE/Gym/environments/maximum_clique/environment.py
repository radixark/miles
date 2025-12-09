import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MaximumClique_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The graph contains the following undirected edges:
{edges}

Your task is to select a subset of vertices `v1, v2, ..., vk` such that:
- 0 ≤ v1, v2, ..., vk < {N} and all selected vertices are **distinct**.
- The selected vertices form a **clique** — that is, **every pair** of distinct selected vertices is connected by **at least one edge**.
- Your goal is to **maximize** the number of selected vertices k.

**Output Format:**
Your final answer should be a single line containing the selected vertex indices `v1, v2, ..., vk`, separated by **spaces**.
Example: `0 2 3` (do **NOT** include the backticks or quotes); this means the selected clique has size k = 3, with vertices 0, 2, and 3.
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MaximumClique_Environment instance.
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
        assert N >= 2, "N should be greater than or equal to 2"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        edges = self.parameter["edges"] = random.sample([(u, v) for u in range(N) for v in range(u + 1, N)], int(edge_density * N * (N - 1) / 2))
        random.shuffle(edges)
        
        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"


        adjacent = [0] * N
        for u, v in edges :
            adjacent[u] |= 1 << v
            adjacent[v] |= 1 << u
        self.parameter["reference_answer"] = []
        clique = []

        def DFS(u : int, allowed_set : int) -> None :
            if len(clique) + (N - u) <= len(self.parameter["reference_answer"]) :
                return
            if u == N :
                assert len(clique) > len(self.parameter["reference_answer"])
                self.parameter["reference_answer"] = clique.copy()
            if allowed_set & (1 << u) :
                clique.append(u)
                DFS(u + 1, allowed_set & adjacent[u])
                clique.pop()
            DFS(u + 1, allowed_set)
        DFS(0, (1 << N) - 1)

        self.parameter["gold_answer"] = len(self.parameter["reference_answer"])
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["reference_answer"]))
            
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
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

            clique = processed_result
            if len(clique) != len(set(clique)) :
                return self.rewards["invalid_solution"]
            for vertex in clique :
                if not (0 <= vertex < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
            edges = set(map(tuple, self.parameter["edges"]))
            for u in clique :
                for v in clique :
                    if u < v :
                        if (u, v) not in edges :
                            return self.rewards["invalid_solution"]
            
            gold, answer = self.parameter["gold_answer"], len(clique)
            assert answer <= gold, "answer should be less than or equal to gold"
            
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]