import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class TreeMaximumVisitedVertex_Environment(VerifiableEnvironment) : # https://www.luogu.com.cn/problem/P3412
    prompt_template = \
r"""You are given a **tree** with {N} vertices labeled from 0 to {N_minus_1}. The tree has the following {N_minus_1} undirected edges:
{edges}

Starting from vertex 0, find a path of length {M} (i.e., consisting of exactly {M} edges) that **maximizes the number of distinct vertices visited at least once**. At each step, you can move to any adjacent vertex; you may revisit vertices in the path. Output {M} + 1 integers (space-separated) representing the labels of the vertices visited along the path, starting from vertex 0."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the TreeMaximumVisitedVertex_Environment instance.
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

        M = self.parameter["M"] = random.randint(2, 2 * (N - 1) - 1)

        edges = self.parameter["edges"] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v))
        random.shuffle(edges)

        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)) == N - 1


        # Adjacency list of size N
        graph = [[] for _ in range(N)]
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)

        # Compute the maximum depth (in nodes) from root 0
        visited = [False] * N
        max_depth = 0

        def dfs(u, depth):
            nonlocal max_depth
            visited[u] = True
            # Update global max_depth
            max_depth = max(max_depth, depth)
            for v in graph[u]:
                if not visited[v]:
                    dfs(v, depth + 1)

        # Perform DFS from node 0, initial depth = 1
        # Use a mutable container to allow assignment in nested scope
        # (Alternatively, declare max_depth as global)
        max_depth = 0
        dfs(0, 1)

        # mx - 1 is the length of the longest path (in edges) from 0
        longest_path_edges = max_depth - 1
        if M <= longest_path_edges:
            # Can only move down the main path
            result = M + 1
        else:
            # Extra moves allow visiting off-path nodes, two steps per new node
            extra = M - longest_path_edges
            result = max_depth + extra // 2
            # Cannot exceed total nodes N
            result = min(N, result)

        self.parameter["gold_answer"] = result


    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
            M = self.parameter["M"],
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List[int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                if not answer_array :
                    return None
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
            if len(path) != self.parameter["M"] + 1 :
                return self.rewards["invalid_solution"]
            if not all(0 <= vertex < self.parameter["N"] for vertex in path) :
                return self.rewards["invalid_solution"]
            if path[0] != 0 :
                return self.rewards["invalid_solution"]
            
            edges = {(u, v) for u, v in self.parameter["edges"]}
            if not all((min(s, t), max(s, t)) in edges for s, t in zip(path, path[1 :])) :
                return self.rewards["invalid_solution"]
            
            answer, gold = len(set(path)), self.parameter["gold_answer"]
            assert 0 < answer <= gold
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]