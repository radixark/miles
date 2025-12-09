import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MinimumSpanningTreeCounting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning an undirected edge **connecting vertex u to vertex v with weight w**:
{edges}

Consider a subset of edges `T = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)]` such that:
- k = {N_minus_1} (i.e., you select exactly {N_minus_1} edges),
- The selected edges form a **spanning tree** — that is, they connect all {N} vertices without forming any cycles,
- The total weight `w_1 + w_2 + ... + w_k` is **minimized** among all such spanning trees (so it is called a minimum spanning tree).

Please compute **the number of such minimum spanning trees** modulo {MOD}."""


    def __init__(self,
                 MAX_MOD : int = 10000,
                 weight_range_divisor : int = 10,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MinimumSpanningTreeCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.weight_range_divisor = weight_range_divisor
        assert self.weight_range_divisor > 0, "weight_range_divisor should be greater than 0"

        self.MAX_MOD = MAX_MOD
        assert self.MAX_MOD > 1, "MAX_MOD should be greater than 1"

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        assert "edge_ratio" in self.parameter, "edge_ratio is required in parameter"
        edge_ratio = self.parameter["edge_ratio"]

        weight_range = max(1, int(edge_ratio * N / self.weight_range_divisor)) + 1

        edges = self.parameter["edges"] = []

        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v, random.randint(1, weight_range)))
        
        num_edges = int(edge_ratio * N)
        if len(edges) < num_edges :
            remaining_edges = list(set((u, v) for u in range(N) for v in range(u + 1, N)) - set((u, v) for u, v, w in edges))
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for u, v in remaining_edges :
                edges.append((u, v, random.randint(1, weight_range)))
        random.shuffle(edges)

        for u, v, w in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, w in edges)), "edges should be unique"

        P = self.parameter["MOD"] = random.randint(2, self.MAX_MOD)


        def find(parent, x):
            # path compression
            if parent[x] != x:
                parent[x] = find(parent, parent[x])
            return parent[x]

        def union(parent, a, b):
            # simple union
            ra = find(parent, a)
            rb = find(parent, b)
            if ra != rb:
                parent[rb] = ra

        def det_mod(mat, mod):
            """
            Compute determinant of mat (n x n) modulo mod using
            a gcd-based elimination that avoids division by non-invertible elements.
            """
            n = len(mat)
            f = 1
            tp = 1
            # ensure entries are in [0, mod)
            for i in range(n):
                for j in range(n):
                    mat[i][j] %= mod

            for i in range(n):
                # eliminate below mat[i][i]
                for j in range(i+1, n):
                    a = mat[i][i]
                    b = mat[j][i]
                    while b:
                        t = a // b
                        a, b = b, a - t*b
                        # row_i = row_i - t * row_j  (from column i onward)
                        for k in range(i, n):
                            mat[i][k] = (mat[i][k] - t * mat[j][k]) % mod
                        # swap row_i and row_j (from column i onward)
                        for k in range(i, n):
                            mat[i][k], mat[j][k] = mat[j][k], mat[i][k]
                        f = -f
                if mat[i][i] % mod == 0:
                    return 0
                tp = tp * (mat[i][i] % mod) % mod

            res = f * tp % mod
            return res if res >= 0 else res + mod

        def count_mst():
            edges = self.parameter["edges"].copy()
            M = len(edges)

            # sort by weight
            edges.sort(key=lambda x: x[2])

            # initialize DSU
            parent = list(range(N))
            ans = 1
            i = 0

            # process groups of equal-weight edges
            while i < M:
                w = edges[i][2]
                j = i
                while j < M and edges[j][2] == w:
                    j += 1
                group = edges[i:j]

                # build the multigraph on current DSU components
                adj_count = {}  # (u, v) -> number of parallel edges
                nodes = set()
                for u, v, _ in group:
                    ru = find(parent, u)
                    rv = find(parent, v)
                    if ru != rv:
                        nodes.add(ru)
                        nodes.add(rv)
                        adj_count[(ru, rv)] = adj_count.get((ru, rv), 0) + 1
                        adj_count[(rv, ru)] = adj_count.get((rv, ru), 0) + 1

                # find connected components in this subgraph
                visited = set()
                for u in nodes:
                    if u in visited:
                        continue
                    # BFS/DFS to collect one component
                    stack = [u]
                    comp = []
                    visited.add(u)
                    while stack:
                        x = stack.pop()
                        comp.append(x)
                        # look at neighbors of x
                        for (a, b), cnt in adj_count.items():
                            if a == x and b not in visited:
                                visited.add(b)
                                stack.append(b)

                    t = len(comp)
                    if t > 1:
                        m = t - 1
                        mat = [[0] * m for _ in range(m)]
                        for xi in range(m):
                            ni = comp[xi]
                            # degree of ni within comp
                            deg = 0
                            for nj in comp:
                                deg += adj_count.get((ni, nj), 0)
                            deg %= P
                            mat[xi][xi] = deg
                            # off-diagonals
                            for yj in range(m):
                                if xi != yj:
                                    nj = comp[yj]
                                    mat[xi][yj] = (- adj_count.get((ni, nj), 0)) % P

                        # multiply in the number of spanning trees of this component
                        ans = ans * det_mod(mat, P) % P

                # unite the DSU by all useful edges in this group
                for u, v, _ in group:
                    ru = find(parent, u)
                    rv = find(parent, v)
                    if ru != rv:
                        union(parent, ru, rv)

                i = j

            # check if the graph is connected
            roots = {find(parent, x) for x in range(N)}
            if len(roots) != 1:
                return 0
            else:
                return ans
        
        self.parameter["reference_answer"] = count_mst()
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {}, {})".format(u, v, w) for u, v, w in self.parameter["edges"]),
            MOD = self.parameter["MOD"],
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
            if not (0 <= processed_result < self.parameter["MOD"]) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]