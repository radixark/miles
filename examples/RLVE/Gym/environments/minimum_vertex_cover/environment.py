import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Minimum_VertexCover_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices labeled from 0 to {N_minus_1}. The graph contains the following undirected edges:
{edges}

Each vertex has a cost, given as a list `C` of length {N}, where `C[i]` is the cost of vertex i:
{C}

Your task is to select a set of distinct vertices x_1, x_2, ..., x_k (you determine k), such that every edge in the graph has at least one endpoint in the selected set; that is, for every edge (u, v), at least one of u or v must be included.
Try your best to minimize the total cost: C[x_1] + C[x_2] + ... + C[x_k].

**Output Format:** Your final answer should be a single line containing the selected vertices in any order, separated by spaces.
Example: `0 1 {N_minus_1}` (do **NOT** include quotes or backticks)."""

    def __init__(self,
                 cost_range : int = 10,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Minimum_DominatingSet_Environment instance.
        """
        super().__init__(**kwargs)

        self.cost_range = cost_range
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
        assert N >= 2, "N should be greater than or equal to 1"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 < edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"
        assert int(edge_density * N * (N - 1) / 2) > 0, "edge_density should be large enough to generate at least one edge"

        edges = self.parameter["edges"] = random.sample([(u, v) for u in range(N) for v in range(u + 1, N)], int(edge_density * N * (N - 1) / 2))
        random.shuffle(edges)
        assert len(edges)
        
        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"

        C = self.parameter["C"] = [random.randint(1, self.cost_range) for vertex in range(N)]


        adjacent = [0] * N
        for u, v in edges :
            adjacent[u] |= 1 << v
            adjacent[v] |= 1 << u

        self.parameter["reference_answer"] = list(range(N))
        self.parameter["gold_answer"] = sum(C)

        selected = []
        def DFS(u : int, not_selected : int, requiring : int, sumC : int) -> None :
            assert (not_selected & requiring) == 0
            if sumC >= self.parameter["gold_answer"] :
                return
            if u == N :
                assert sumC < self.parameter["gold_answer"]
                self.parameter["reference_answer"], self.parameter["gold_answer"] = selected.copy(), sumC
                return
            
            if not (requiring & (1 << u)) :
                if not (not_selected & adjacent[u]) :
                    DFS(u + 1, not_selected | (1 << u), requiring | adjacent[u], sumC)
            
            selected.append(u)
            DFS(u + 1, not_selected, requiring, sumC + C[u])
            selected.pop()
        DFS(0, 0, 0, 0)

        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["reference_answer"]))
        assert self.parameter["gold_answer"] > 0

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
            C = "\n".join("C[{}]={}".format(i, Ci) for i, Ci in enumerate(self.parameter["C"])),
        )
    
    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            selected_vertices = processed_result

            if not all(0 <= vertex < self.parameter["N"] for vertex in selected_vertices) :
                return self.rewards["invalid_solution"]
            if len(selected_vertices) != len(set(selected_vertices)) :
                return self.rewards["invalid_solution"]
            selected_vertices = set(selected_vertices)

            for u, v in self.parameter["edges"] :
                if (u not in selected_vertices) and (v not in selected_vertices) :
                    return self.rewards["invalid_solution"]

            answer = sum(self.parameter["C"][u] for u in selected_vertices)
            gold = self.parameter["gold_answer"]
            assert gold <= answer, "gold should be less than or equal to answer"

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]