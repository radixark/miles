import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Vertex_KCenter_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected connected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning an undirected edge **connecting vertex `u` to vertex `v` with weight `w`**:
{edges}

Please select a set of {K} distinct vertices. Try your best to minimize the largest distance of any vertex in the graph to its closest vertex in the selected set; the distance between two vertices `u` and `v` is defined as the sum of the weights of the edges in the **shortest path** connecting them.

**Output Format:** Your final answer should be a single line containing the selected {K} vertices in any order, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Vertex_KCenter_Environment instance.
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

        K = self.parameter["K"] = random.randint(1, N - 1)

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
            edges.append((u, v, random.randint(1, N)))
        
        num_edges = int(edge_density * N * (N - 1) / 2)
        if len(edges) < num_edges :
            remaining_edges = list(set((u, v) for u in range(N) for v in range(u + 1, N)) - set((u, v) for u, v, w in edges))
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for u, v in remaining_edges :
                edges.append((u, v, random.randint(1, N)))
        random.shuffle(edges)

        Floyd = self.parameter["Floyd"] = [[N * N] * N for _ in range(N)]
        for i in range(N) :
            Floyd[i][i] = 0

        for u, v, w in edges :
            assert 0 <= u < v < N
            Floyd[u][v] = Floyd[v][u] = w
        assert len(edges) == len(set((u, v) for u, v, w in edges)), "edges should be unique"

        for k in range(N) :
            for i in range(N) :
                for j in range(N) :
                    val = Floyd[i][k] + Floyd[k][j]
                    if val < Floyd[i][j] :
                        Floyd[i][j] = val
        

        self.parameter["reference_answer"], self.parameter["gold_answer"] = None, N * N
        solution, solution_dist = [], [N * N] * N
        def DFS(u : int) -> None :
            nonlocal solution, solution_dist

            if len(solution) + (N - u) < K :
                return
            if N == u :
                assert len(solution) == K, "solution should have exactly K elements"
                current_answer = max(solution_dist)
                if current_answer < self.parameter["gold_answer"] :
                    self.parameter["reference_answer"], self.parameter["gold_answer"] = solution.copy(), current_answer
                return
            
            DFS(u + 1)
            if len(solution) < K :
                solution.append(u)
                cache_solution_dist = solution_dist.copy()
                for v in range(N) :
                    solution_dist[v] = min(solution_dist[v], Floyd[u][v])
                DFS(u + 1)
                solution_dist = cache_solution_dist
                solution.pop()
        DFS(0)
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["reference_answer"]))
        assert self.parameter["gold_answer"] > 0
        

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
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            selected_vertices = processed_result

            if len(selected_vertices) != len(set(selected_vertices)) :
                return self.rewards["invalid_solution"]
            if len(selected_vertices) != self.parameter["K"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= u < self.parameter["N"] for u in selected_vertices) :
                return self.rewards["invalid_solution"]

            answer = 0
            for u in range(self.parameter["N"]) :
                dist = self.parameter["Floyd"][u][selected_vertices[0]]
                for selected_vertex in selected_vertices[1 :] :
                    dist = min(dist, self.parameter["Floyd"][u][selected_vertex])
                answer = max(answer, dist)
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