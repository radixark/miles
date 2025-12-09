import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinimumSpanningTree_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning an undirected edge **connecting vertex `u` to vertex `v` with weight `w`**:
{edges}

Your task is to select a subset of edges `T = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)]` such that:
- k = {N} - 1 = {N_minus_1} (i.e., you select exactly {N_minus_1} edges).
- The selected edges form a **spanning tree** — that is, they connect all {N} vertices without forming any cycles.
- Your goal is to **minimize** the total weight of the selected edges: `w_1 + w_2 + ... + w_k`.

**Output Format:**
Your final answer should be a single line containing the endpoints of the selected edges in order: `u_1 v_1 u_2 v_2 ... u_k v_k`, separated by **spaces**.
Example: `0 1 1 2 2 3` (do **NOT** include the backticks or quotes); this means the spanning tree includes the edges `(0, 1, w_1)`, `(1, 2, w_2)`, and `(2, 3, w_3)` (assuming 4 vertices in total)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinimumSpanningTree_Environment instance.
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

        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v, random.randint(1, max(1, int(edge_density * N * (N - 1) / 2)))))
        
        num_edges = int(edge_density * N * (N - 1) / 2)
        if len(edges) < num_edges :
            remaining_edges = list(set((u, v) for u in range(N) for v in range(u + 1, N)) - set((u, v) for u, v, w in edges))
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for u, v in remaining_edges :
                edges.append((u, v, random.randint(1, max(1, int(edge_density * N * (N - 1) / 2)))))
        random.shuffle(edges)

        for u, v, w in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, w in edges)), "edges should be unique"


        G = networkx.Graph()
        G.add_weighted_edges_from(edges)
        mst = networkx.minimum_spanning_tree(G)
        self.parameter["reference_answer"] = " ".join("{} {}".format(u, v) for u, v in mst.edges())
        self.parameter["gold_answer"] = sum(mst[u][v]["weight"] for u, v in mst.edges())

        assert self.parameter["gold_answer"] > 0, "The gold answer should be greater than 0"
    
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

            mst = processed_result
            if len(mst) % 2 != 0 :
                return self.rewards["wrong_format"]
            mst = [(mst[i], mst[i + 1]) for i in range(0, len(mst), 2)]
            
            if len(mst) != self.parameter["N"] - 1 :
                return self.rewards["invalid_solution"]
            if not ((set(u for u, v in mst) | set(v for u, v in mst)) == set(range(self.parameter["N"]))) :
                return self.rewards["invalid_solution"]

            subgraph = networkx.Graph()
            edge2weight = {(u, v) : w for u, v, w in self.parameter["edges"]}            
            answer_weight = 0
            for u, v in mst :
                u, v = min(u, v), max(u, v)
                if (u, v) not in edge2weight :
                    return self.rewards["invalid_solution"]
                answer_weight += edge2weight[(u, v)]
                subgraph.add_edge(u, v)
            if not networkx.is_connected(subgraph) :
                return self.rewards["invalid_solution"]
            assert networkx.is_tree(subgraph), "The answer should be a tree as it has N - 1 edges and is connected"
            
            assert self.parameter["gold_answer"] <= answer_weight, "answer_weight should be greater than or equal to gold_answer"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((self.parameter["gold_answer"] / answer_weight) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == answer_weight)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]