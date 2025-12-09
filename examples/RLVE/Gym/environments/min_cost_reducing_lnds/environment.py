import random
from collections import deque
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinCostReducingLNDS_Environment(VerifiableEnvironment) : # Submitted to https://www.luogu.com.cn/problem/P3308
    prompt_template = \
r"""You are given two arrays A and B, both of length {N}:
A: {A}  
B: {B}
You may erase any (distinct) elements from A. When you erase element A[i], you must pay a cost of B[i]. Please reduce the length of the **longest non-decreasing subsequence** (not necessarily contiguous) of A by **at least 1**, while minimizing the total cost of the erased elements.
**Output Format:** Output a single line containing the **indices** of the elements you choose to erase, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the MinCostReducingLNDS_Environment instance.
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

        self.parameter["A"] = [random.randint(1, max(1, N * 2)) for _ in range(N)]
        self.parameter["B"] = [random.randint(1, N) for _ in range(N)]

        
        class Edge:
            __slots__ = ('to','rev','cap','orig')
            def __init__(self, to, rev, cap):
                self.to = to
                self.rev = rev
                self.cap = cap
                self.orig = cap

        def add_edge(u, v, c):
            """Add edge u->v with capacity c, and reverse edge."""
            adj[u].append(Edge(v, len(adj[v]), c))
            adj[v].append(Edge(u, len(adj[u]) - 1, 0))

        def bfs_level():
            """Build level graph from source; return True if sink reachable."""
            for i in range(V):
                level[i] = -1
            q = deque([SRC])
            level[SRC] = 0
            while q:
                u = q.popleft()
                for e in adj[u]:
                    if e.cap > 0 and level[e.to] < 0:
                        level[e.to] = level[u] + 1
                        q.append(e.to)
            return level[SINK] >= 0

        def dfs_flow(u, f):
            """DFS in level graph; push up to f units, return actual pushed."""
            if u == SINK:
                return f
            for i in range(ptr[u], len(adj[u])):
                e = adj[u][i]
                if e.cap > 0 and level[e.to] == level[u] + 1:
                    pushed = dfs_flow(e.to, min(f, e.cap))
                    if pushed:
                        e.cap -= pushed
                        adj[e.to][e.rev].cap += pushed
                        return pushed
                ptr[u] += 1
            return 0

        def dinic():
            """Run Dinic to exhaustion; return total flow."""
            flow = 0
            while bfs_level():
                ptr[:] = [0] * V
                while True:
                    pushed = dfs_flow(SRC, INF)
                    if not pushed:
                        break
                    flow += pushed
            return flow

        def reachable(u, t):
            """Simple BFS on current residual graph to check if t reachable from u."""
            vis = [False] * V
            dq = deque([u])
            vis[u] = True
            while dq:
                x = dq.popleft()
                if x == t:
                    return True
                for e in adj[x]:
                    if e.cap > 0 and not vis[e.to]:
                        vis[e.to] = True
                        dq.append(e.to)
            return False

        A = [0] + self.parameter["A"].copy()
        B = [0] + self.parameter["B"].copy()
        C = [0] + list(range(1, N + 1))

        V = 2 * N + 2
        SRC, SINK = 0, V - 1
        adj = [[] for _ in range(V)]

        # 1) Node-split edges; record their positions for later removal
        id_info = [None] * (N + 1)
        for i in range(1, N + 1):
            u, v = i, N + i
            idx_u = len(adj[u])
            idx_v = len(adj[v])
            adj[u].append(Edge(v, idx_v, B[i]))
            adj[v].append(Edge(u, idx_u, 0))
            id_info[i] = (u, idx_u, v, idx_v)

        # 2) Compute dp[i] = LIS ending at i
        dp = [0] * (N + 1)
        dp[0] = 0
        for i in range(1, N + 1):
            best = 1
            for j in range(1, i):
                if A[j] <= A[i] and dp[j] + 1 > best:
                    best = dp[j] + 1
            dp[i] = best

        K = max(dp[1:])
        self.parameter["original_lnds_length"] = K

        # 3) Add DAG edges with infinite capacity = INF
        S = sum(B[1:]) + 1
        INF = S

        for i in range(1, N + 1):
            # from source to level-1 nodes
            if dp[i] == 1:
                add_edge(SRC, i, INF)
            # from level-K nodes to sink
            if dp[i] == K:
                add_edge(N + i, SINK, INF)
            # between intermediate levels
            for j in range(1, i):
                if A[j] <= A[i] and dp[j] + 1 == dp[i]:
                    add_edge(N + j, i, INF)

        # 4) Initial max-flow = minimal total cost
        level = [-1] * V
        ptr = [0] * V
        INF = S
        flow = dinic()
        # flow is the minimal cost S
        assert flow > 0, "The flow should be greater than 0"
        self.parameter["gold_answer"] = flow

        # 5) Greedy extract lexicographically smallest C-sorted cut
        vc = sorted((C[i], i) for i in range(1, N + 1))
        ans = []
        remaining_flow = flow

        for _, idx in vc:
            # if idx.in can't reach idx.out in residual, it's essential
            if not reachable(idx, N + idx):
                ans.append(idx)
                # permanently remove its split edge
                u, iu, v, iv = id_info[idx]
                e1 = adj[u][iu]
                e2 = adj[v][iv]
                e1.orig = 0
                e2.orig = 0
                # reset all capacities to orig
                for u0 in range(V):
                    for e in adj[u0]:
                        e.cap = e.orig
                # recompute flow on the reduced graph
                level = [-1] * V
                ptr = [0] * V
                remaining_flow = dinic()
                if remaining_flow == 0:
                    break

        # 6) Output M and the sorted positions
        ans = [i - 1 for i in ans]
        assert self.parameter["gold_answer"] == sum(self.parameter["B"][i] for i in ans), \
            f"Gold answer {self.parameter['gold_answer']} does not match computed cost {sum(self.parameter['B'][i] for i in ans)}"
        self.parameter["reference_answer"] = " ".join(map(str, ans))
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
            B = " ".join("B[{}]={}".format(i, Bi) for i, Bi in enumerate(self.parameter["B"])),
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
            
            erased = [False] * self.parameter["N"]
            for i in processed_result :
                if not (0 <= i < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                if erased[i] :
                    return self.rewards["invalid_solution"]
                erased[i] = True
            
            newA = [Ai for i, Ai in enumerate(self.parameter["A"]) if not erased[i]]
            F = [0] * len(newA)
            for i, Ai in enumerate(newA) :
                F[i] = 1
                for j, Aj in enumerate(newA[: i]) :
                    if Aj <= Ai :
                        F[i] = max(F[i], F[j] + 1)
            
            assert (max(F) if F else 0) <= self.parameter["original_lnds_length"]
            if (max(F) if F else 0) == self.parameter["original_lnds_length"] :
                return self.rewards["unsuccessful_solution"]

            answer, gold = sum(self.parameter["B"][i] for i in processed_result), self.parameter["gold_answer"]
            assert gold <= answer, "Gold answer should be less than or equal to the answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * int(gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]