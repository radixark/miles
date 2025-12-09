import queue
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class DynDynamite_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3523
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices, labeled from `0` to `{N_minus_1}`. It contains the following {N_minus_1} undirected edges. Each edge is represented as a tuple `(u, v)`, meaning there is an undirected edge **connecting vertex `u` to vertex `v`**:
{edges}

You are also given a list of key vertices: {key_vertices}
Please select exactly {M} vertices (from all {N} vertices) to serve as **centers**. Your goal is to **minimize the maximum distance** (measured in number of edges) from any key vertex to its nearest selected center.
Output format: A single line containing the {M} selected centers, separated by spaces."""


    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the DynDynamite_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

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

        key_vertices = self.parameter["key_vertices"] = random.sample(range(N), random.randint(2, N))
        M = self.parameter["M"] = random.randint(1, len(key_vertices) - 1)


        d = [0] * N
        for key_vertex in key_vertices :
            d[key_vertex] = 1

        # Build adjacency list (0-indexed)
        adj = [[] for _ in range(N)]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)

        # Build a parent array and a preorder traversal 'order'
        parent = [-1] * N
        order = []
        stack = [0]
        parent[0] = -1
        while stack:
            x = stack.pop()
            order.append(x)
            for v in adj[x]:
                if v == parent[x]:
                    continue
                parent[v] = x
                stack.append(v)

        # Sentinels for DP
        NEG_INF = -(N + 1)
        INF = N + 1

        # Given a time limit t, compute the minimum number of ignitions needed
        def needed(t: int) -> int:
            f = [NEG_INF] * N
            g = [INF] * N
            cnt = 0

            # Process in reverse preorder (children before parent)
            for x in reversed(order):
                # If an existing ignition in the subtree covers
                # the nearest uncovered bomb within t, discard it
                if f[x] + g[x] <= t:
                    f[x] = NEG_INF

                # If there's an uncovered bomb here (g[x]>t) and
                # this room has a bomb, place an ignition here
                if g[x] > t and d[x] == 1:
                    if f[x] < 0:
                        f[x] = 0

                # If an ignition at distance exactly t reaches here,
                # "use it up" and count it
                if f[x] == t:
                    f[x] = NEG_INF
                    g[x] = 0
                    cnt += 1

                # Propagate distances up to the parent
                p = parent[x]
                if p != -1:
                    # furthest ignition distance
                    val_f = f[x] + 1
                    if val_f > f[p]:
                        f[p] = val_f
                    # nearest bomb distance
                    val_g = g[x] + 1
                    if val_g < g[p]:
                        g[p] = val_g

            # If there's still an ignition reaching the root, count it
            if f[0] >= 0:
                cnt += 1
            return cnt

        # Binary search on the answer t in [0, N]
        l, r = 0, N
        while l < r:
            mid = (l + r) // 2
            if needed(mid) <= M:
                r = mid
            else:
                l = mid + 1

        self.parameter["gold_answer"] = l
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("{} {}".format(u, v) for u, v in self.parameter["edges"]),
            key_vertices = " ".join(map(str, self.parameter["key_vertices"])),
            M = self.parameter["M"],
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

            if len(processed_result) != self.parameter["M"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= vertex < self.parameter["N"] for vertex in processed_result) :
                return self.rewards["invalid_solution"]
            
            adj = [[] for _ in range(self.parameter["N"])]
            for a, b in self.parameter["edges"] :
                adj[a].append(b)
                adj[b].append(a)
            Q = queue.Queue()
            distance = [None] * self.parameter["N"]
            for start in processed_result :
                distance[start] = 0
                Q.put(start)
            while not Q.empty() :
                u = Q.get()
                for v in adj[u] :
                    if distance[v] is None :
                        distance[v] = distance[u] + 1
                        Q.put(v)
            
            answer, gold = max(distance[u] for u in self.parameter["key_vertices"]), self.parameter["gold_answer"]
            assert 0 < gold <= answer, "gold should be greater than 0 and less than or equal to answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]