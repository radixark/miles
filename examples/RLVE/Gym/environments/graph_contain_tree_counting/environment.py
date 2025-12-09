import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class GraphContainTreeCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3349
    prompt_template = \
r"""You are given an **undirected graph** G and a **tree** T, each with {N} vertices labeled from `0` to `{N_minus_1}`.

- Graph G has the following undirected edge set E1:
{G_edges}

- Tree T has the following undirected edge set E2:
{T_edges}

Please compute the number of **bijections** `p` (i.e., permutations) from the vertices of T to the vertices of G such that: for every edge `(u, v)` in E2, the edge `(p(u), p(v))` exists in E1.

**Output Format:** A single integer representing the number of valid bijections."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the GraphContainTreeCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"


        edges = self.parameter["T_edges"] = []
        permutation = list(range(N))
        random.shuffle(permutation)
        for index, vertex in enumerate(permutation) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutation[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v))
        random.shuffle(edges)
        
        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)) == N - 1


        edges = self.parameter["G_edges"] = []
        random.shuffle(permutation)
        for u, v in self.parameter["T_edges"] :
            u, v = permutation[u], permutation[v]
            if u > v :
                u, v = v, u
            edges.append((u, v))
        
        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]

        num_edges = int(edge_density * N * (N - 1) / 2)
        if len(edges) < num_edges :
            remaining_edges = list(set((u, v) for u in range(N) for v in range(u + 1, N)) - set(edges))
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            edges += remaining_edges
        random.shuffle(edges)

        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"
        edges = None


        G = [[False] * N for _ in range(N)]
        for u, v in self.parameter["G_edges"]:
            G[u][v] = G[v][u] = True

        # Remaining tree edges
        ADJ = [[] for _ in range(N)]
        for u, v in self.parameter["T_edges"]:
            ADJ[u].append(v)
            ADJ[v].append(u)

        # vis[i] indicates whether original vertex i is selected in the current subset
        vis = [False] * N

        # f[u][x] will hold the number of ways to map the subtree of the current tree,
        # rooted at u, when u maps to original-graph vertex x
        f = [[0] * N for _ in range(N)]

        ans = 0  # final answer accumulator

        def dfs(u, parent, whi):
            """
            Perform the DP on the tree:
            For each node u, and for each selected original-graph vertex x in whi,
            compute f[u][x] = product over children v of (sum over y in whi & G[x][y] of f[v][y]).
            """
            for v in ADJ[u]:
                if v == parent:
                    continue
                dfs(v, u, whi)

            # Now compute f[u][x] for each x in the current subset
            for x in whi:
                f[u][x] = 1
                for v in ADJ[u]:
                    if v == parent:
                        continue
                    total = 0
                    for y in whi:
                        if G[x][y]:
                            total += f[v][y]
                    f[u][x] *= total

        def solve():
            """
            For the current subset of original-graph vertices (marked by vis),
            collect them in whi[], run the tree-DP rooted at 0,
            then add or subtract from ans according to the parity of N - |whi|.
            """
            nonlocal ans
            whi = [i for i in range(N) if vis[i]]
            dfs(0, -1, whi)

            # Inclusion–exclusion: subtract if (N - |whi|) is odd, else add
            if (N - len(whi)) & 1:
                for x in whi:
                    ans -= f[0][x]
            else:
                for x in whi:
                    ans += f[0][x]

        def enumerate_subsets(dep=0):
            """
            Recursively enumerate all subsets of {0,1,...,N-1} by toggling vis[dep].
            When dep == N, process the current subset.
            """
            if dep == N:
                solve()
                return
            # Exclude dep
            vis[dep] = False
            enumerate_subsets(dep + 1)
            # Include dep
            vis[dep] = True
            enumerate_subsets(dep + 1)

        # Kick off the subset enumeration and DP
        enumerate_subsets()

        # Output the final count
        assert ans > 0
        self.parameter["reference_answer"] = ans


    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            G_edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["G_edges"]),
            T_edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["T_edges"]),
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
            if processed_result <= 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]