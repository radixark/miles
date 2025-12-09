import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class DegreeFixed_SpanningTree_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v)`, meaning an undirected edge **connecting vertex `u` to vertex `v`**:
{edges}

Your task is to select a subset of edges `T = [(u_1, v_1), (u_2, v_2), ..., (u_k, v_k)]` such that:
- The selected edges form a **spanning tree** — that is, they connect all {N} vertices without forming any cycles.
- Each vertex `i` has a **fixed degree** of `d_i`, meaning it must be connected to exactly `d_i` edges in the selected subset: {degrees}

**Output Format:**
Your final answer should be a single line containing the endpoints of the selected edges in order: `u_1 v_1 u_2 v_2 ... u_k v_k`, separated by **spaces**.
Example: `0 1 1 2 2 3` (do **NOT** include the backticks or quotes); this means the spanning tree includes the edges `(0, 1)`, `(1, 2)`, and `(2, 3)` (assuming 4 vertices in total), where the degrees of 0, 1, 2, and 3 are 1, 2, 2, and 1 respectively."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
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

        degrees = self.parameter["degrees"] = [0] * N
        edges = self.parameter["edges"] = []

        self.parameter["reference_answer"] = []

        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            self.parameter["reference_answer"].append("{} {}".format(u, v))
            u, v = min(u, v), max(u, v)
            edges.append((u, v))
            degrees[u] += 1
            degrees[v] += 1
        
        self.parameter["reference_answer"] = " ".join(self.parameter["reference_answer"])
        
        num_edges = int(edge_density * N * (N - 1) / 2)
        if len(edges) < num_edges :
            remaining_edges = list(set((u, v) for u in range(N) for v in range(u + 1, N)) - set(edges))
            edges += random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
        random.shuffle(edges)

        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
            degrees = ", ".join("d_{}={}".format(i, degree) for i, degree in enumerate(self.parameter["degrees"])),
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

            st = processed_result
            if len(st) % 2 != 0 :
                return self.rewards["wrong_format"]
            st = [(st[i], st[i + 1]) for i in range(0, len(st), 2)]
            
            if len(st) != self.parameter["N"] - 1 :
                return self.rewards["invalid_solution"]
            if not ((set(u for u, v in st) | set(v for u, v in st)) == set(range(self.parameter["N"]))) :
                return self.rewards["invalid_solution"]
            
            degrees = [0] * self.parameter["N"]

            subgraph = networkx.Graph()
            edges = set(map(tuple, self.parameter["edges"]))
            for u, v in st :
                u, v = min(u, v), max(u, v)
                if (u, v) not in edges :
                    return self.rewards["invalid_solution"]
                subgraph.add_edge(u, v)
                degrees[u] += 1
                degrees[v] += 1
            if not networkx.is_connected(subgraph) :
                return self.rewards["invalid_solution"]
            assert networkx.is_tree(subgraph), "The answer should be a tree as it has N - 1 edges and is connected"
            
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                satisfied = sum(int(d_answer == d_gold) for d_answer, d_gold in zip(degrees, self.parameter["degrees"]))
                return self.rewards["rewarding_weight"] * ((satisfied / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * all(d_answer == d_gold for d_answer, d_gold in zip(degrees, self.parameter["degrees"]))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]