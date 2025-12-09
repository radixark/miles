import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinCostTreeCoverage_Environment(VerifiableEnvironment) : # Submitted to https://www.luogu.com.cn/problem/P3267
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from `0` to `{N_minus_1}`. The tree contains {N_minus_1} undirected edges. Each edge is represented as a tuple `(u, v)`, meaning there is an undirected edge **connecting vertex `u` to vertex `v`**:
{edges}

You may select any subset of vertices. When a vertex `u` is selected, it **covers** all vertices that are reachable from `u` by a path containing at most {D} edges (i.e., within distance ≤ {D} in terms of edge count). You are required to cover the following vertices: {covered_vertices}
Each selected vertex `u` incurs a cost of `W[u]`. The cost array is: {W}
Try your best to **minimize the total cost** of the selected vertices while ensuring all required vertices are covered.

**Output Format:** A single line containing the selected vertex indices in any order, separated by **spaces**."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the MinCostTreeCoverage_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "unsuccessful_solution" : unsuccessful_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        edges = self.parameter["edges"] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        depths = [None] * N
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                depths[vertex] = 0
                continue
            u, v = vertex, random.choice(permutations[: index])
            depths[u] = depths[v] + 1
            u, v = min(u, v), max(u, v)
            edges.append((u, v))
        random.shuffle(edges)

        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)) == N - 1

        covered_vertices = self.parameter["covered_vertices"] = random.sample(range(N), k = random.randint(1, N))

        D = self.parameter["D"] = random.randint(1, max(1, max(depths[covered_vertex] for covered_vertex in covered_vertices) // 2))

        W = self.parameter["W"] = [random.randint(1, N) for _ in range(N)]


        important = [False] * N
        for x in covered_vertices:
            important[x] = True             # 0-index

        A = [[] for _ in range(N)]              # adjacency list
        for u, v in edges:                     # 0-index
            A[u].append(v)
            A[v].append(u)

        # ---------- constants & DP tables ---------------------------------------
        K = D                                   # alias used below
        INF = sum(W) + 1                        # far larger than any legal answer

        # dp[u][i]  : u *not* yet covered by an ancestor guard.
        # fdp[u][i] : u *already* covered by an ancestor guard.
        # ‘i’ is the distance (0 … K) from u to the closest guard in u’s subtree
        dp  = [[INF] * (K + 1) for _ in range(N)]
        fdp = [[INF] * (K + 1) for _ in range(N)]

        for i in range(N):
            dp[i][K] = W[i]                     # place a guard on i
            if important[i]:
                fdp[i][0] = 0                   # covered by ancestor is fine
            else:
                dp[i][0] = 0                    # no guard needed (not important)

        # ---------- build parent / post-order without recursion ------------------
        parent   = [-1] * N
        children = [[] for _ in range(N)]
        order    = []                           # pre-order → reversed ⇒ post-order

        stack = [0]
        parent[0] = 0                           # root sentinel
        while stack:
            u = stack.pop()
            order.append(u)
            for v in A[u]:
                if parent[v] == -1:
                    parent[v] = u
                    children[u].append(v)
                    stack.append(v)

        # ---------- DP merge -----------------------------------------------------
        for u in reversed(order):               # post-order
            for v in children[u]:
                # prefix minima helper arrays (length K+1)
                tru = [0] * (K + 1)
                trv = [0] * (K + 1)

                tru[0] = min(dp[u])             # min cost in u-subtree
                for i in range(1, K + 1):
                    tru[i] = min(tru[i - 1], fdp[u][i - 1])

                trv[0] = min(dp[v])             # min cost in v-subtree
                for i in range(1, K + 1):
                    trv[i] = min(trv[i - 1], fdp[v][i - 1])

                new_dp  = [0] * (K + 1)
                new_fdp = [0] * (K + 1)

                # --- update dp[u] (u not yet covered by ancestor) ---------------
                for i in range(K):              # 0 … K-1
                    new_dp[i] = min(dp[u][i] + trv[i],
                                    dp[v][i + 1] + tru[i + 1])
                    if new_dp[i] > INF:
                        new_dp[i] = INF
                new_dp[K] = dp[u][K] + trv[K]
                if new_dp[K] > INF:
                    new_dp[K] = INF

                # --- update fdp[u] (u already covered by ancestor) --------------
                new_fdp[0] = fdp[u][0] + trv[0]
                if new_fdp[0] > INF:
                    new_fdp[0] = INF
                for i in range(1, K + 1):
                    new_fdp[i] = min(fdp[u][i] + trv[i],
                                    fdp[v][i - 1] + tru[i])
                    if new_fdp[i] > INF:
                        new_fdp[i] = INF

                dp[u]  = new_dp
                fdp[u] = new_fdp

        # ---------- answer -------------------------------------------------------
        self.parameter["gold_answer"] = min(dp[0])
        assert self.parameter["gold_answer"] > 0, "Gold answer should be greater than 0"
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("{} {}".format(u, v) for u, v in self.parameter["edges"]),
            covered_vertices = " ".join(map(str, self.parameter["covered_vertices"])),
            D = self.parameter["D"],
            W = " ".join("W[{}]={}".format(i, Wi) for i, Wi in enumerate(self.parameter["W"])),
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

            if len(processed_result) != len(set(processed_result)) :
                return self.rewards["invalid_solution"]
            
            answer, gold = 0, self.parameter["gold_answer"]

            adjacency_list = [[] for _ in range(self.parameter["N"])]
            for u, v in self.parameter["edges"] :
                adjacency_list[u].append(v)
                adjacency_list[v].append(u)
            
            covered = [False] * self.parameter["N"]
            for vertex in processed_result :
                if not (0 <= vertex < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                answer += self.parameter["W"][vertex]
                visited = [False] * self.parameter["N"]
                visited[vertex] = True
                stack = [(vertex, 0)]
                while stack :
                    u, d = stack.pop()
                    covered[u] = True
                    if d == self.parameter["D"] :
                        continue
                    for v in adjacency_list[u] :
                        if not visited[v] :
                            visited[v] = True
                            stack.append((v, d + 1))

            if not all(covered[covered_vertex] for covered_vertex in self.parameter["covered_vertices"]) :
                return self.rewards["unsuccessful_solution"]
            assert gold <= answer, "Gold answer should be less than or equal to the answer"

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise ValueError("Invalid rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]