import random
from collections import deque
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Tree_DistanceEqualTriad_Counting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3565
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices, labeled from `1` to `{N}`. It contains the following {N_minus_1} undirected edges:
{edges}

Please compute the number of three-vertex sets (a triad of vertices A, B, and C such that 1 ≤ A < B < C ≤ {N}) for which the **pairwise distances** are all equal — that is, the distance between A and B, between A and C, and between B and C are all the same. The distance between two vertices is the number of edges on the shortest path connecting them."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the Tree_DistanceEqualTriad_Counting_Environment instance.
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


        adjacency = [[] for _ in range(N+1)]
        for a, b in edges:
            adjacency[a].append(b)
            adjacency[b].append(a)

        ans = 0

        # For each candidate center c, we look at its branches (one per neighbor).
        # In each branch we BFS to record how many nodes lie at each distance d from c.
        # Then for each distance d we have counts [c1, c2, ..., ck] across branches,
        # and the number of ways to pick one node in three distinct branches all at that
        # same distance is the 3rd elementary symmetric sum:
        #    e3 = sum_{i<j<k} ci*cj*ck = (S1^3 - 3 S1 S2 + 2 S3)/6,
        # where S1 = sum ci, S2 = sum ci^2, S3 = sum ci^3.

        for c in range(1, N+1):
            if len(adjacency[c]) < 3:
                continue  # need at least 3 branches

            visited = [False] * (N+1)
            visited[c] = True

            branch_counts = []
            max_depth = 0

            # BFS each branch separately, marking visited to avoid overlap
            for nbr in adjacency[c]:
                if visited[nbr]:
                    continue
                visited[nbr] = True
                q = deque([(nbr, 1)])
                local = []  # local[d] = number of nodes at distance d in this branch
                while q:
                    u, d = q.popleft()
                    # ensure local is long enough
                    if d >= len(local):
                        local.extend([0] * (d - len(local) + 1))
                    local[d] += 1
                    if d > max_depth:
                        max_depth = d
                    for w in adjacency[u]:
                        if not visited[w]:
                            visited[w] = True
                            q.append((w, d+1))
                branch_counts.append(local)

            b = len(branch_counts)
            if b < 3:
                continue

            # for each possible distance t, compute the 3‐way product sum
            for t in range(1, max_depth+1):
                S1 = S2 = S3 = 0
                for f in branch_counts:
                    cnt = f[t] if t < len(f) else 0
                    S1 += cnt
                    S2 += cnt*cnt
                    S3 += cnt*cnt*cnt
                # elementary symmetric sum of order 3
                e3 = (S1*S1*S1 - 3*S1*S2 + 2*S3) // 6
                ans += e3

        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
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
            if processed_result < 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                if processed_result == 0 :
                    return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == 0)
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]