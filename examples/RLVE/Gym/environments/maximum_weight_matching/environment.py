import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MaximumWeightMatching_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning an undirected edge **connecting vertex `u` to vertex `v` with weight `w`**:
{edges}

Your task is to select a subset of edges `S = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)]` such that:
- Each selected edge must exist in the graph.
- **Each vertex appears in at most one edge** in the set `S` — in other words, no two edges in `S` share a vertex.
- Your goal is to **maximize** the total weight of the selected edges `w_1 + w_2 + ... + w_k`.

**Output Format:**
Your final answer should be a single line containing the endpoints of the selected edges in order: `u_1 v_1 u_2 v_2 ... u_k v_k`, separated by **spaces**.  
Example: `0 1 3 4` (do **NOT** include the backticks or quotes); this means k = 2 edges are selected: `(0, 1, w_1)` and `(3, 4, w_2)`.
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MaximumWeightMatching_Environment instance.
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

        edges = self.parameter["edges"] = random.sample([(u, v, random.randint(1, N)) for u in range(N) for v in range(u + 1, N)], int(edge_density * N * (N - 1) / 2))
        random.shuffle(edges)
        
        assert len(edges) == len(set((u, v) for u, v, w in edges)), "edges should be unique"
        for u, v, w in edges :
            assert 0 <= u < v < N

        G = networkx.Graph()
        G.add_weighted_edges_from(edges)
        matching = networkx.max_weight_matching(G, maxcardinality = False)
        self.parameter["reference_answer"] = " ".join("{} {}".format(u, v) for u, v in matching)

        edge2weight = {(u, v) : w for u, v, w in edges}
        self.parameter["gold_weight"] = sum(edge2weight[(min(u, v), max(u, v))] for u, v in matching)
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {}, {})".format(u, v, w) for u, v, w in self.parameter["edges"]),
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

            matches = processed_result
            if len(matches) % 2 != 0 :
                return self.rewards["wrong_format"]
            matches = [(matches[i], matches[i + 1]) for i in range(0, len(matches), 2)]

            if not (len(set(u for u, v in matches) | set(v for u, v in matches)) == len(matches) * 2) :
                return self.rewards["invalid_solution"]
            edge2weight = {(u, v) : w for u, v, w in self.parameter["edges"]}
            answer_weight = 0
            for u, v in matches :
                u, v = min(u, v), max(u, v)
                if (u, v) not in edge2weight :
                    return self.rewards["invalid_solution"]
                answer_weight += edge2weight[(u, v)]
            assert answer_weight <= self.parameter["gold_weight"], "answer_weight should be less than or equal to gold_weight"
            
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer_weight / self.parameter["gold_weight"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer_weight == self.parameter["gold_weight"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]