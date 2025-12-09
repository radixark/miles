import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class FixedOneEdgeNum_SpanningTree_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3623
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v, w)` (`w` is either 0 or 1), meaning an undirected edge **connecting vertex `u` to vertex `v` with weight `w`**:
{edges}

Please select a subset of edges `T = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)]` such that:
- k = {N_minus_1} (i.e., you select exactly {N_minus_1} edges),
- The selected edges form a **spanning tree** — that is, they connect all {N} vertices without forming any cycles,
- There are exactly {K} edges with weight 1 in the selected edges,

**Output Format:** Your final answer should be a single line containing the endpoints of the selected edges in order: `u_1 v_1 u_2 v_2 ... u_k v_k`, separated by **spaces**.
Example: `0 1 1 2 2 3` (do **NOT** include the backticks or quotes); this means the spanning tree includes the edges `(0, 1, w_1)`, `(1, 2, w_2)`, and `(2, 3, w_3)` (assuming 4 vertices in total)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, wrong_solution : float = 0.0, correct_solution : float = +1.0,
                 **kwargs) :
        """
        Initialize the FixedOneEdgeNum_SpanningTree_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "wrong_solution" : wrong_solution,
            "correct_solution" : correct_solution,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        assert "edge_ratio" in self.parameter, "edge_ratio is required in parameter"
        edge_ratio = self.parameter["edge_ratio"]

        edges = self.parameter["edges"] = []

        permutations = list(range(N))
        random.shuffle(permutations)
        one_probability = random.random()
        self.parameter["K"], self.parameter["reference_answer"] = 0, []
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            self.parameter["reference_answer"].append("{} {}".format(u, v))
            u, v, w = min(u, v), max(u, v), int(random.random() < one_probability)
            edges.append((u, v, w))
            self.parameter["K"] += w
        self.parameter["reference_answer"] = " ".join(self.parameter["reference_answer"])
        
        num_edges = int(edge_ratio * N)
        if len(edges) < num_edges :
            remaining_edges = list(set((u, v) for u in range(N) for v in range(u + 1, N)) - set((u, v) for u, v, w in edges))
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            one_probability = random.random()
            for u, v in remaining_edges :
                edges.append((u, v, int(random.random() < one_probability)))
        random.shuffle(edges)

        for u, v, w in edges :
            assert 0 <= u < v < N
            assert w in (0, 1), "edge weight should be either 0 or 1"
        assert len(edges) == len(set((u, v) for u, v, w in edges)), "edges should be unique"
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            K = self.parameter["K"],
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
            
            if answer_weight != self.parameter["K"] :
                return self.rewards["wrong_solution"]
            else :
                return self.rewards["correct_solution"]
        else :
            return self.rewards["wrong_format"]