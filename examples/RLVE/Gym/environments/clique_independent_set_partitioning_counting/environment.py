import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Clique_IndependentSet_Partitioning_Counting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3513
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The graph contains the following undirected edges:
{edges}

Please output the number of ways to partition the vertices into two **non-empty** sets S and T such that:
- S is a **clique** (i.e., every pair of distinct vertices in S is connected by an edge),
- T is an **independent set** (i.e., no pair of distinct vertices in T is connected by an edge),
- S and T are **disjoint** (i.e., S ∩ T = ∅)."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the Clique_IndependentSet_Partitioning_Counting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        clique = random.sample(range(N), random.randint(2, N - 1))
        independent_set = list(set(range(N)) - set(clique))
        edges = self.parameter["edges"] = []
        for u in clique :
            for v in clique :
                if u < v :
                    edges.append((u, v))
        edges += random.sample([(min(u, v), max(u, v)) for u in clique for v in independent_set], random.randint(0, len(clique) * len(independent_set)))
        random.shuffle(edges)

        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"


        flg = [ [False]*N for _ in range(N) ]
        for u, v in edges :
            flg[u][v]= flg[v][u] = True

        # 2-SAT implication graph on 2*N nodes: 
        #  0..N-1   == X_i (i is in the support group)
        #  N..2N-1  == ¬X_i (i is in the conspiracy group)
        dfn      = [0] * (2*N)
        low      = [0] * (2*N)
        in_stack = [False] * (2*N)
        col      = [0] * (2*N)
        stack    = []
        tot, colid = 0, 0

        def tarjan(u):
            nonlocal tot, colid
            tot += 1
            dfn[u] = low[u] = tot
            in_stack[u] = True
            stack.append(u)

            pos = u % N
            for i in range(N):
                if i == pos:
                    continue
                v = -1
                # if u represents ¬X_pos (i.e. u>=N) and pos knows i,
                #    then add implication ¬X_pos → X_i
                if u >= N and flg[pos][i]:
                    v = i
                # if u represents X_pos (u<N) and pos doesn't know i,
                #    then X_pos → ¬X_i
                if u < N and not flg[pos][i]:
                    v = i + N

                if v != -1:
                    if dfn[v] == 0:
                        tarjan(v)
                        low[u] = min(low[u], low[v])
                    elif in_stack[v]:
                        low[u] = min(low[u], dfn[v])

            if low[u] == dfn[u]:
                colid += 1
                while True:
                    x = stack.pop()
                    in_stack[x] = False
                    col[x] = colid
                    if x == u:
                        break

        # Run Tarjan on all nodes
        for u in range(2*N):
            if dfn[u] == 0:
                tarjan(u)

        # Check unsatisfiable: X_i and ¬X_i in same SCC
        for i in range(N):
            if col[i] == col[i+N]:
                assert False, "The problem is unsatisfiable: X_i and ¬X_i are in the same strongly connected component."
                return

        # Build one satisfying assignment:
        # if col[X_i] < col[¬X_i], put i in S1, else in S2
        S1 = []
        S2 = []
        for i in range(N):
            if col[i] < col[i+N]:
                S1.append(i)
            else:
                S2.append(i)

        # Precompute for each person how many "cross-edges" they have
        deg = [0]*N
        # For any i in S1, count how many j in S2 that i knows
        for i in S1:
            for j in S2:
                if flg[i][j]:
                    deg[i] += 1
        # For any j in S2, count how many i in S1 that j does NOT know
        for j in S2:
            for i in S1:
                if not flg[i][j]:
                    deg[j] += 1

        # Now count all valid partitions reachable by swapping at most one
        # member between S1 and S2 (including the “no swap” case).
        ans = 0
        cnt1 = len(S1)
        cnt2 = len(S2)

        # Use None as the “dummy” to represent “no element swapped”
        S1d = [None] + S1
        S2d = [None] + S2

        for x in S1d:
            for y in S2d:
                # new sizes after removing x (if any) from S1 and adding y (if any)
                C1 = cnt1 - (1 if x is not None else 0) + (1 if y is not None else 0)
                C2 = cnt2 - (1 if y is not None else 0) + (1 if x is not None else 0)
                if C1 == 0 or C2 == 0:
                    continue

                v1 = deg[x] if x is not None else 0
                v2 = deg[y] if y is not None else 0

                # if we swapped two real people, adjust the double-counted edge
                if x is not None and y is not None:
                    if flg[x][y]:
                        v1 -= 1
                    else:
                        v2 -= 1

                if v1 == 0 and v2 == 0:
                    ans += 1

        assert ans > 0
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
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
            if processed_result < 0 :
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