import random
import networkx
from collections import deque
from typing import Optional, Tuple
from Gym.environment import VerifiableEnvironment


class TreeChangeOneEdgeDiameter_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3596
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from `1` to `{N}`. The tree contains the following edges:
{edges}

You may remove one edge from the tree and add a new edge (possibly the same edge) such that the resulting graph is still a tree. Your goal is to {maximize_or_minimize} the diameter of the resulting tree; the **diameter** of a tree is defined as the number of edges on the longest path between any two vertices.

**Output Format:** Output four integers `u1 v1 u2 v2` (do NOT include the backticks or quotes), separated by spaces, where:
- `(u1, v1)` is the edge to be removed
- `(u2, v2)` is the edge to be added"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5,
                 rewarding_strategy_min : str = "(gold/answer)^beta", rewarding_weight_min : float = +1.0, rewarding_beta_min : float = 5.0,
                 rewarding_strategy_max : str = "(answer/gold)^beta", rewarding_weight_max : float = +1.0, rewarding_beta_max : float = 5.0,
                 **kwargs) :
        """
        Initialize the TreeChangeOneEdgeDiameter_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy_min": rewarding_strategy_min,
            "rewarding_weight_min": rewarding_weight_min,
            "rewarding_beta_min": rewarding_beta_min,
            "rewarding_strategy_max": rewarding_strategy_max,
            "rewarding_weight_max": rewarding_weight_max,
            "rewarding_beta_max": rewarding_beta_max
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

        self.parameter["minimize_or_maximize"] = random.choice(["minimize", "maximize"])


        A = [[] for _ in range(N + 1)]
        for u, v in edges:
            A[u].append(v)
            A[v].append(u)

        def get_diameter(start, skip_u=None, skip_v=None):
            # First BFS (or DFS) to find one end of the diameter
            dist = [-1] * (N + 1)
            dist[start] = 0
            q = deque([start])
            far = start
            while q:
                u = q.popleft()
                for v in A[u]:
                    if skip_u is not None and ((u == skip_u and v == skip_v) or (u == skip_v and v == skip_u)):
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                        if dist[v] > dist[far]:
                            far = v
            # Second BFS from that end to find the other end and record parents
            P = [-1] * (N + 1)  # 2) capitalized variable P for parent
            dist2 = [-1] * (N + 1)
            dist2[far] = 0
            q = deque([far])
            far2 = far
            while q:
                u = q.popleft()
                for v in A[u]:
                    if skip_u is not None and ((u == skip_u and v == skip_v) or (u == skip_v and v == skip_u)):
                        continue
                    if dist2[v] == -1:
                        dist2[v] = dist2[u] + 1
                        P[v] = u
                        q.append(v)
                        if dist2[v] > dist2[far2]:
                            far2 = v
            # Reconstruct the diameter path
            D = []  # 2) capitalized D for diameter list
            u = far2
            while u != -1:
                D.append(u)
                u = P[u]
            return D

        def get_farthest(start, skip_u=None, skip_v=None):
            dist = [-1] * (N + 1)
            dist[start] = 0
            q = deque([start])
            far = start
            while q:
                u = q.popleft()
                for v in A[u]:
                    if skip_u is not None and ((u == skip_u and v == skip_v) or (u == skip_v and v == skip_u)):
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                        if dist[v] > dist[far]:
                            far = v
            return far

        # Original diameter
        D = get_diameter(1)
        InDiameter = [False] * (N + 1)
        for u in D:
            InDiameter[u] = True

        # f[u]: longest chain from u into a subtree off the diameter
        # g[u]: diameter within u's off-diameter subtree
        f = [0] * (N + 1)
        g = [0] * (N + 1)

        def tree_dp(u, p):
            for v in A[u]:
                if v == p:
                    continue
                tree_dp(v, u)
                if InDiameter[v]:
                    continue
                old_f = f[u]
                # update g[u]
                g[u] = max(g[u], g[v], f[v] + 1 + old_f)
                # update f[u]
                f[u] = max(old_f, f[v] + 1)

        tree_dp(D[0], 0)

        L = len(D)
        # prefix DP
        pref = [0] * L
        cur = 0
        for i in range(L):
            u = D[i]
            if i == 0:
                pref[i] = max(0, g[u], cur + f[u])
            else:
                pref[i] = max(pref[i - 1], g[u], cur + f[u])
            cur = max(cur + 1, f[u] + 1)

        # 5) INF computed from input
        INF = N + 5
        kmin = INF
        kmax = -INF
        x1min = y1min = x2min = y2min = None
        x1max = y1max = x2max = y2max = None

        # suffix DP + find best removal for min/max
        R = 0
        cur = 0
        for i in range(L - 1, 0, -1):
            u = D[i]
            R = max(R, g[u], cur + f[u])
            cur = max(cur + 1, f[u] + 1)
            left = pref[i - 1]
            # candidate for minimal new diameter
            cand_min = max(left, R, (R + 1)//2 + (left + 1)//2 + 1)
            if cand_min < kmin:
                kmin = cand_min
                x1min, y1min = u, D[i - 1]
            # candidate for maximal new diameter
            if R + 1 + left > kmax:
                kmax = R + 1 + left
                x1max, y1max = u, D[i - 1]

        # also consider removing a single off-diameter branch edge for max
        for u in D:
            for v in A[u]:
                if not InDiameter[v]:
                    if L + g[v] > kmax:
                        kmax = L + g[v]
                        x1max, y1max = u, v

        # find the new-edge endpoints for the minimal case
        D1 = get_diameter(x1min, x1min, y1min)
        x2min = D1[(len(D1) - 1) // 2]
        D2 = get_diameter(y1min, x1min, y1min)
        y2min = D2[(len(D2) - 1) // 2]

        # and for the maximal case
        x2max = get_farthest(x1max, x1max, y1max)
        y2max = get_farthest(y1max, x1max, y1max)

        # output
        if self.parameter["minimize_or_maximize"] == "minimize" :
            self.parameter["gold_answer"] = kmin
            self.parameter["reference_answer"] = "{} {} {} {}".format(x1min, y1min, x2min, y2min)
        elif self.parameter["minimize_or_maximize"] == "maximize" :
            self.parameter["gold_answer"] = kmax
            self.parameter["reference_answer"] = "{} {} {} {}".format(x1max, y1max, x2max, y2max)
        else :
            assert False, "minimize_or_maximize should be either 'minimize' or 'maximize'"
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
            maximize_or_minimize = self.parameter["minimize_or_maximize"],
        )


    def _process(self, answer : Optional[str]) -> Optional[Tuple[int, int, int, int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                u1, v1, u2, v2 = map(int, answer.split())
                return u1, v1, u2, v2
            except :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            u1, v1, u2, v2 = processed_result

            edges = [(u, v) for u, v in self.parameter["edges"] if (u, v) != (min(u1, v1), max(u1, v1))]
            if len(edges) != self.parameter["N"] - 2 :
                assert len(edges) == self.parameter["N"] - 1, "There should be exactly N-1 edges in the tree"
                return self.rewards["invalid_solution"]
            if not (1 <= u2 <= self.parameter["N"] and 1 <= v2 <= self.parameter["N"] and u2 != v2 and (min(u2, v2), max(u2, v2)) not in edges) :
                return self.rewards["invalid_solution"]
            edges.append((u2, v2))

            G = networkx.Graph()
            G.add_edges_from(edges)
            if not networkx.is_tree(G) :
                return self.rewards["invalid_solution"]
            assert set([u for u, v in edges] + [v for u, v in edges]) == set(range(1, self.parameter["N"] + 1)), "All vertices should be present in the tree"
            
            answer, gold = networkx.diameter(G), self.parameter["gold_answer"]
            if self.parameter["minimize_or_maximize"] == "minimize" :
                assert 0 < gold <= answer, "For minimization, answer should be greater than 0 and at least as large as the gold answer"
                if self.rewards["rewarding_strategy_min"] == "(gold/answer)^beta" :
                    return self.rewards["rewarding_weight_min"] * ((gold / answer) ** self.rewards["rewarding_beta_min"])
                elif self.rewards["rewarding_strategy_min"] == "gold=answer" :
                    return self.rewards["rewarding_weight_min"] * (gold == answer)
                else :
                    raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy_min"]))
            elif self.parameter["minimize_or_maximize"] == "maximize" :
                assert 0 < answer <= gold, "For maximization, answer should be greater than 0 and at most as large as the gold answer"
                if self.rewards["rewarding_strategy_max"] == "(answer/gold)^beta" :
                    return self.rewards["rewarding_weight_max"] * ((answer / gold) ** self.rewards["rewarding_beta_max"])
                elif self.rewards["rewarding_strategy_max"] == "gold=answer" :
                    return self.rewards["rewarding_weight_max"] * (gold == answer)
                else :
                    raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy_max"]))
            else :
                assert False, "minimize_or_maximize should be either 'minimize' or 'maximize'"
        else :
            return self.rewards["wrong_format"]