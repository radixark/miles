import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinimumDirectedSpanningTree_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a **directed graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following directed edges. Each edge is represented as a tuple `(s, t, w)`, meaning a directed edge **from vertex `s` to vertex `t` with weight `w`**:
{edges}

Your task is to select a subset of edges `T = [(s_1, t_1, w_1), (s_2, t_2, w_2), ..., (s_k, t_k, w_k)]` such that:
- k = {N} - 1 = {N_minus_1} (i.e., you select exactly {N_minus_1} edges).
- The selected edges form a **spanning arborescence rooted at vertex {root}** — meaning:
  - All vertices are reachable from vertex `{root}`.
  - Each vertex other than `{root}` has exactly one incoming edge.
  - The selected edges form no cycles.
- Your goal is to **minimize** the total weight of the selected edges: `w_1 + w_2 + ... + w_k`.

**Output Format:**
Your final answer should be a single line containing the endpoints of the selected edges in order: `s_1 t_1 s_2 t_2 ... s_k t_k`, separated by **spaces**.
Example: `0 1 0 2 2 3` (do **NOT** include the backticks or quotes); this means the arborescence includes edges `(0, 1)`, `(0, 2)`, and `(2, 3)` (assuming 4 vertices in total and root = 0)."""


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

        while True :
            edges = self.parameter["edges"] = []

            permutations = list(range(N))
            random.shuffle(permutations)
            for index, vertex in enumerate(permutations) :
                if index == 0 :
                    continue
                t, s = vertex, random.choice(permutations[: index])
                edges.append((s, t, random.randint(1, max(1, int(edge_density * N * (N - 1))))))
            root = self.parameter["root"] = permutations[0]
            
            num_edges = int(edge_density * N * (N - 1))
            if len(edges) < num_edges :
                remaining_edges = list(set((s, t) for s in range(N) for t in range(N) if s != t) - set((s, t) for s, t, w in edges))
                remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
                for s, t in remaining_edges :
                    edges.append((s, t, random.randint(1, max(1, int(edge_density * N * (N - 1))))))
            random.shuffle(edges)

            for s, t, w in edges :
                assert 0 <= s < N and 0 <= t < N, "s and t should be in range [0, N)"
                assert s != t
            assert len(edges) == len(set((s, t) for s, t, w in edges)), "edges should be unique"

            try :
                G = networkx.DiGraph()
                G.add_weighted_edges_from(edges + [(self.parameter["N"], root, 0)])
                msa = networkx.minimum_spanning_arborescence(G)
                self.parameter["reference_answer"] = " ".join("{} {}".format(s, t) for s, t in msa.edges() if (s, t) != (self.parameter["N"], root))
                self.parameter["gold_answer"] = sum(msa[s][t]["weight"] for s, t in msa.edges())
                assert self.parameter["gold_answer"] > 0, "The gold answer should be greater than 0"
                break
            except : # There might a bug in networkx.minimum_spanning_arborescence
                continue
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {}, {})".format(s, t, w) for s, t, w in self.parameter["edges"]),
            root = self.parameter["root"],
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

            msa = processed_result
            if len(msa) % 2 != 0 :
                return self.rewards["wrong_format"]
            msa = [(msa[i], msa[i + 1]) for i in range(0, len(msa), 2)]
            
            if len(msa) != self.parameter["N"] - 1 :
                return self.rewards["invalid_solution"]
            if not ((set(s for s, t in msa) | set(t for s, t in msa)) == set(range(self.parameter["N"]))) :
                return self.rewards["invalid_solution"]

            adjacent_list = [[] for s in range(self.parameter["N"])]
            for s, t in msa :
                assert 0 <= s < self.parameter["N"] and 0 <= t < self.parameter["N"], "s and t should be in range [0, N)"
                if s == t :
                    return self.rewards["invalid_solution"]
                adjacent_list[s].append(t)
            
            visited = [False] * self.parameter["N"]
            def DFS(vertex : int) -> bool :
                for neighbor in adjacent_list[vertex] :
                    if visited[neighbor] :
                        return False
                    visited[neighbor] = True
                    if not DFS(neighbor) :
                        return False
                return True
            visited[self.parameter["root"]] = True
            if not DFS(self.parameter["root"]) :
                return self.rewards["invalid_solution"]
            if not all(visited) :
                return self.rewards["invalid_solution"]
            
            G = networkx.DiGraph()
            G.add_nodes_from(range(self.parameter["N"] + 1))
            G.add_edges_from(msa + [(self.parameter["N"], self.parameter["root"])])
            assert networkx.is_arborescence(G)
            
            edges = {(s, t) : w for s, t, w in self.parameter["edges"]}
            answer_weight = 0
            for s, t in msa :
                if (s, t) not in edges :
                    return self.rewards["invalid_solution"]
                answer_weight += edges[(s, t)]
            assert self.parameter["gold_answer"] <= answer_weight, "answer_weight should be greater than or equal to gold_answer"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((self.parameter["gold_answer"] / answer_weight) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == answer_weight)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]