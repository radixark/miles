import random
import networkx
from typing import Optional
from Gym.environment import VerifiableEnvironment


class LinkBeads_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3647
    prompt_template = \
r"""You are given a connected undirected graph with {N} nodes labeled from 0 to {N_minus_1}, connected by {N_minus_1} undirected edges (so this is a tree). Each edge is represented as a tuple `(u, v, w)`, meaning there is an undirected edge **connecting vertex `u` to vertex `v` with weight `w`:
{edges}

These edges are the result of a sequence of operations, each either:
- `Append(x, v)`: Add a new node `x` and connect it to an existing node `v` with a **red edge**.
- `Insert(x, u, v)`: Remove the **red edge** between nodes `u` and `v`, and add **two blue edges** - one from `u` to `x` and one from `x` to `v`.

After all operations, the final tree is given (as above), but the **edge colors are unknown**. Your task is to determine the **maximum total length of blue edges** that could exist in any valid sequence of operations that produces the given graph.

**Output Format:** A single integer — the maximum possible total length of blue edges."""

    def __init__(self,
                 wrong_format : float = -1.0, wrong_answer : float = 0.0, correct_answer : float = +1.0,
                 **kwargs) :
        """
        Initialize the LinkBeads_Environment intance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_answer" : wrong_answer,
            "correct_answer" : correct_answer,
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
            edges.append((u, v, random.randint(1, N)))
        random.shuffle(edges)
        
        for u, v, w in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, w in edges)) == N - 1

        tree = networkx.Graph()
        tree.add_weighted_edges_from(edges)
        assert networkx.is_tree(tree)


        class MultiSetMax:
            def __init__(self):
                # we store negatives so that heapq (a min‐heap) behaves like a max‐heap
                self._add = []
                self._rem = []

            def insert(self, x):
                # push the delta
                import heapq
                heapq.heappush(self._add, -x)

            def remove(self, x):
                # lazily remove by pushing into the “removal” heap
                import heapq
                heapq.heappush(self._rem, -x)

            def _clean(self):
                import heapq
                # pop from both heaps while their tops match
                while self._add and self._rem and self._add[0] == self._rem[0]:
                    heapq.heappop(self._add)
                    heapq.heappop(self._rem)

            def get_max(self):
                self._clean()
                if not self._add:
                    return None
                return -self._add[0]

        adj = [[] for _ in range(N)]
        total_weight = 0

        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
            total_weight += w

        # A safe “-infinity” based on input size:
        NEG_INF = -(total_weight + 5)

        dp0 = [0] * N        # dp0[x] == dp[x][0]
        dp1 = [0] * N        # dp1[x] == dp[x][1]
        summ = [0] * N       # summ[x] accumulates the sum of best child contributions
        st = [MultiSetMax() for _ in range(N)]

        # First DFS: compute dp0, dp1, summ and fill each st[x]
        def dfs(x, parent):
            for y, w in adj[x]:
                if y == parent:
                    continue
                dfs(y, x)
                # matching the C++: 
                # v1 = max(dp[y][0], dp[y][1] + w)
                # v2 = dp[y][0] + w
                v1 = dp0[y]
                if dp1[y] + w > v1:
                    v1 = dp1[y] + w
                v2 = dp0[y] + w

                summ[x] += v1
                st[x].insert(v2 - v1)

            dp0[x] = summ[x]
            m = st[x].get_max()
            dp1[x] = summ[x] + m if m is not None else NEG_INF

        ans = 0

        # Second DFS: rerooting to consider every node as “root”
        def dfs0(x, parent):
            nonlocal ans
            # we can only count blue‐score when parent‐edge is red → dp0[x]
            if dp0[x] > ans:
                ans = dp0[x]

            for y, w in adj[x]:
                if y == parent:
                    continue

                # Backup all mutable state for x and y
                bx0, bx1, bsx = dp0[x], dp1[x], summ[x]
                by0, by1, bsy = dp0[y], dp1[y], summ[y]

                # Remove y’s contribution from x
                v1y = dp0[y] if dp0[y] >= dp1[y] + w else dp1[y] + w
                v2y = dp0[y] + w
                delta_xy = v2y - v1y

                st[x].remove(delta_xy)
                summ[x] -= v1y
                dp0[x] = summ[x]
                mx = st[x].get_max()
                dp1[x] = summ[x] + mx if mx is not None else NEG_INF

                # Add x’s contribution to y as if we’d “rerooted” the tree at y
                v1x = dp0[x] if dp0[x] >= dp1[x] + w else dp1[x] + w
                v2x = dp0[x] + w
                delta_yx = v2x - v1x

                summ[y] += v1x
                st[y].insert(delta_yx)
                dp0[y] = summ[y]
                my = st[y].get_max()
                dp1[y] = summ[y] + my if my is not None else NEG_INF

                # Recurse
                dfs0(y, x)

                # Restore states
                dp0[x], dp1[x], summ[x] = bx0, bx1, bsx
                st[x].insert(delta_xy)      # undo the removal

                dp0[y], dp1[y], summ[y] = by0, by1, bsy
                st[y].remove(delta_yx)      # undo the insertion

        dfs(0, -1)
        dfs0(0, -1)
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {}, {})".format(u, v, w) for u, v, w in self.parameter["edges"]),
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