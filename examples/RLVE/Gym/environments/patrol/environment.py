import random
from typing import Optional
from collections import deque
from Gym.environment import VerifiableEnvironment


class Patrol_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3629
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from `1` to `{N}`. It contains the following {N_minus_1} undirected edges:
{edges}

You are allowed to add {K} arbitrary edges to the tree. Each added edge can connect any two existing vertices (including possibly the same vertex); it is allowed to be a duplicate of an existing edge. After adding these {K} edges, you must start at vertex `1` (and also end at vertex `1`) and traverse a path that:
- Visits each **original edge at least once**, and
- Visits each **added edge exactly once**.

Please output the **minimum total number of edges traversed** (of course, edges that are traversed multiple times should be counted multiple times) in such a path."""
    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the Patrol_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 4, "N should be greater than or equal to 4"

        edges = self.parameter["edges"] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u + 1, v + 1))  # Convert to 1-based indexing
        random.shuffle(edges)

        for u, v in edges :
            assert 1 <= u < v <= N
        assert len(edges) == len(set(edges)) == N - 1

        K = self.parameter["K"] = random.randint(1, 2)


        # Build adjacency list for the tree
        adj = [[] for _ in range(N + 1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # BFS to find farthest node and distance from a start node
        def bfs(start, record_parent=False):
            dist = [-1] * (N + 1)
            parent = [0] * (N + 1)
            q = deque([start])
            dist[start] = 0
            far_node = start
            maxd = 0
            while q:
                x = q.popleft()
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        q.append(y)
                        if dist[y] > maxd:
                            maxd = dist[y]
                            far_node = y
            if record_parent:
                return far_node, maxd, parent, dist
            return far_node, maxd

        # First BFS from node 1 to find one end of the diameter
        u, _ = bfs(1)
        # Second BFS from u to find the other end, and record parents
        v, L1, parent, _ = bfs(u, record_parent=True)

        # Case K = 1: formula is 2*(N-1) - L1 + 1
        if K == 1:
            result = 2 * (N - 1) - L1 + 1
            self.parameter["reference_answer"] = result
            return

        # For K = 2: mark the nodes on the diameter path
        on_path = [False] * (N + 1)
        node = v
        while node != 0:
            on_path[node] = True
            node = parent[node]

        # Prepare for DP to compute L2 (weighted diameter with diameter edges weight -1)
        d = [0] * (N + 1)
        L2 = [0]

        def dfs(x, p):
            for y in adj[x]:
                if y == p:
                    continue
                dfs(y, x)
                # weight = -1 if edge is on the original diameter, else +1
                w = -1 if on_path[x] and on_path[y] else 1
                # update the maximum combination across two branches
                L2[0] = max(L2[0], d[x] + d[y] + w)
                # update the best single branch length
                d[x] = max(d[x], d[y] + w)

        # Run DP from root = 1
        dfs(1, 0)

        # Final answer for K = 2: 2*N - L1 - L2
        result = 2 * N - L1 - L2[0]
        self.parameter["reference_answer"] = result
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            K = self.parameter["K"],
            edges = "\n".join("{} {}".format(u, v) for u, v in self.parameter["edges"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[int] :
        if answer is not None :
            answer = answer.strip()
            try :
                int_answer = int(answer)
                return int_answer
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]