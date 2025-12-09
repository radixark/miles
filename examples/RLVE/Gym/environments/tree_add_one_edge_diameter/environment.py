import random
import networkx
from typing import Optional, Tuple
from Gym.environment import VerifiableEnvironment


class TreeAddOneEdgeDiameter_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3771
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from `1` to `{N}`. The tree contains the following {N_minus_1} undirected edges, where each tuple `(u, v, w)` represents an edge between vertices `u` and `v` with weight `w`:
{edges}

Let's add **exactly one undirected edge** with weight {L} to the tree. Our goal is to minimize the **longest distance** between any two vertices in the resulting graph. The distance between two vertices is defined as the sum of edge weights along the shortest path connecting them. Output two integers `x y` (do NOT include quotes), separated by a space, indicating the two vertices to which the new edge of weight {L} is added."""
    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5,
                 rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the TreeAddOneEdgeDiameter_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
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
            edges.append((u + 1, v + 1, random.randint(0, N)))  # Convert to 1-based indexing
        random.shuffle(edges)

        for u, v, w in edges :
            assert 1 <= u < v <= N
        assert len(edges) == len(set((u, v) for u, v, w in edges)) == N - 1

        L = self.parameter["L"] = random.randint(0, N)


        NEG_INF = 0

        # Build adjacency list
        e = [[] for _ in range(N+1)]
        for u, v, w in edges:
            e[u].append((v, w))
            e[v].append((u, w))
            NEG_INF -= w + 1

        # 1) Find S: the farthest node from node 1
        dis1 = [0] * (N+1)
        stack = [(1, 0)]
        while stack:
            u, p = stack.pop()
            for v, w in e[u]:
                if v == p:
                    continue
                dis1[v] = dis1[u] + w
                stack.append((v, u))
        S = max(range(1, N+1), key=lambda i: dis1[i])

        # 2) DFS from S to compute distances (dis) and subtree max-distance (mx), plus parent pointers
        dis = [0] * (N+1)
        mx  = [0] * (N+1)
        parent = [0] * (N+1)
        stack2 = [(S, 0, 0)]  # (node, parent, state) state=0: pre, state=1: post
        while stack2:
            u, p, st = stack2.pop()
            if st == 0:
                parent[u] = p
                stack2.append((u, p, 1))
                for v, w in e[u]:
                    if v == p:
                        continue
                    dis[v] = dis[u] + w
                    stack2.append((v, u, 0))
            else:
                mxd = dis[u]
                for v, _ in e[u]:
                    if v == p:
                        continue
                    if mx[v] > mxd:
                        mxd = mx[v]
                mx[u] = mxd

        # 3) Find T: the farthest node from S, and record the original diameter
        T = max(range(1, N+1), key=lambda i: dis[i])
        diam = dis[T]

        # 4) Extract the diameter path from S to T
        p_nodes = []
        u = T
        while True:
            p_nodes.append(u)
            if u == S:
                break
            u = parent[u]
        p_nodes.reverse()
        cnt = len(p_nodes)

        # 5) Compute prefix distances along the path (pre) and branch depths (val)
        pre = [0] * (cnt+2)
        val = [0] * (cnt+2)
        for i in range(1, cnt+1):
            pre[i] = dis[p_nodes[i-1]]
        for i in range(1, cnt+1):
            node = p_nodes[i-1]
            prev_node = p_nodes[i-2] if i > 1     else None
            next_node = p_nodes[i]   if i < cnt else None
            best = 0
            for v, _ in e[node]:
                if v == prev_node or v == next_node:
                    continue
                depth = mx[v] - dis[node]
                if depth > best:
                    best = depth
            val[i] = best

        # 6) Prepare sorted index lists for the two-pointer checks
        p1 = [0] + sorted(range(1, cnt+1), key=lambda i: val[i] + pre[i])
        p2 = [0] + sorted(range(1, cnt+1), key=lambda i: val[i] - pre[i], reverse=True)

        # 7) Feasibility check: can we achieve diameter <= x after adding the new edge?
        def check(x):
            A = B = C = D = NEG_INF
            mx1 = mx2 = NEG_INF
            j = 0

            # First pass: accumulate constraints from violating pairs
            for idx in range(1, cnt+1):
                i_idx = p1[idx]
                while j+1 <= cnt and (val[i_idx] + pre[i_idx] +
                                      val[p2[j+1]] - pre[p2[j+1]] > x):
                    j += 1
                    k = p2[j]
                    c1 = val[k] + pre[k]
                    if c1 > mx1: mx1 = c1
                    c2 = val[k] - pre[k]
                    if c2 > mx2: mx2 = c2

                # Update A, B, C, D
                t = val[i_idx] + pre[i_idx] + mx1
                if t > A: A = t
                t = val[i_idx] - pre[i_idx] + mx1
                if t > B: B = t
                t = val[i_idx] + pre[i_idx] + mx2
                if t > C: C = t
                t = val[i_idx] - pre[i_idx] + mx2
                if t > D: D = t

                # If no pairs violated for all i, it's already feasible
                if idx == cnt and j == 0:
                    return True

            # Adjust constraints by (L - x)
            delta = L - x
            A += delta; B += delta; C += delta; D += delta

            # Second pass: sliding-window ranges
            a, b, c, d = cnt+1, 1, 0, cnt
            for i_idx in range(1, cnt+1):
                while a > 1 and pre[i_idx] + pre[a-1] >= A:
                    a -= 1
                while b <= cnt and -pre[i_idx] + pre[b] < B:
                    b += 1
                while c < cnt and pre[i_idx] - pre[c+1] >= C:
                    c += 1
                while d >= 1 and -pre[i_idx] - pre[d] < D:
                    d -= 1

                left = a if a > b else b
                r1 = c if c < d else d
                right = i_idx-1 if i_idx-1 < r1 else r1
                if left <= right:
                    return True

            return False

        # 8) Binary search for the minimal achievable diameter
        left, right, ans = 0, diam, diam
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {}, {})".format(u, v, w) for u, v, w in self.parameter["edges"]),
            L = self.parameter["L"],
        )


    def _process(self, answer : Optional[str]) -> Optional[Tuple[int, int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                x, y = map(int, answer.split())
                return x, y
            except :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            x, y = processed_result
            if not (1 <= x <= self.parameter["N"] and 1 <= y <= self.parameter["N"]) :
                return self.rewards["invalid_solution"]

            G = networkx.MultiGraph()
            G.add_weighted_edges_from(self.parameter["edges"])
            G.add_edge(x, y, weight = self.parameter["L"])
            answer, gold = max(max(networkx.single_source_dijkstra_path_length(G, u, weight = "weight").values()) for u in G.nodes()), self.parameter["gold_answer"]
            assert 0 <= gold <= answer, "The answer should be at least as large as the gold answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if answer == 0 :
                    assert gold == 0, "gold should be zero if answer is zero"
                    return self.rewards["rewarding_weight"] * 1.0  # Reward for zero answer
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]