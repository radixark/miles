import random
import networkx
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MostComponentTreeRemovingTwoPaths_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3748
    prompt_template = \
r"""You are given a **tree** with {N} vertices labeled from 1 to {N}, where vertex 1 is the **root**. The tree contains the following {N_minus_1} undirected edges:
{edges}

Your task is to choose two paths (each from any vertex to any vertex; a path could be just one single vertex) such that:
- The two paths do **NOT** share any edge (but they can share vertices).
- You remove all vertices on both paths, along with all their adjacent edges.
- After this removal, the remaining structure is a forest. Try your best to **maximize the number of connected components** in the resulting forest.

**Output Format:** A single integer — the maximum number of connected components you can achieve."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MostComponentTreeRemovingTwoPaths instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
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
            u, v = min(u, v) + 1, max(u, v) + 1
            edges.append((u, v))
        random.shuffle(edges)
        
        for u, v in edges :
            assert 1 <= u < v <= N
        assert len(edges) == len(set(edges)) == N - 1

        tree = networkx.Graph()
        tree.add_edges_from(edges)
        assert networkx.is_tree(tree)


        adj = [[] for _ in range(N+1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # build a child-only adjacency by rooting at 1
        visited = [False] * (N+1)
        visited[1] = True
        stack = [1]
        children = [[] for _ in range(N+1)]
        order = []
        while stack:
            u = stack.pop()
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    children[u].append(v)
                    stack.append(v)
        # we no longer need 'adj'
        # do the DP in post-order
        ans = 0
        f0 = [0]*(N+1)
        f1 = [0]*(N+1)
        f2 = [0]*(N+1)
        f3 = [0]*(N+1)
        for u in reversed(order):
            deg_u = len(children[u])
            dp0 = deg_u
            dp1 = 1
            dp2 = deg_u
            dp3 = deg_u
            ret = 0
            off = 1 if u == 1 else 0
            for q in children[u]:
                c0, c1, c2, c3 = f0[q], f1[q], f2[q], f3[q]
                # update global answer
                val = dp3 + c0 - off
                if val > ans: ans = val
                val = dp0 + c3 - off
                if val > ans: ans = val
                val = dp1 + c2
                if val > ans: ans = val
                val = dp1 + c1 - 1
                if val > ans: ans = val
                val = dp2 + c1 - off
                if val > ans: ans = val
                val = dp2 + c2 - off
                if val > ans: ans = val
                # transitions for f1
                if c1 > dp1: dp1 = c1
                if c2 + 1 > dp1: dp1 = c2 + 1
                # transitions for f3
                val = dp0 + c2 - 1
                if val > dp3: dp3 = val
                val = dp0 + c1 - 1
                if val > dp3: dp3 = val
                val = dp2 + c0 - 1
                if val > dp3: dp3 = val
                val = c3 + deg_u - 1
                if val > dp3: dp3 = val
                val = c0 + deg_u + ret - 2
                if val > dp3: dp3 = val
                # transitions for f2
                val = dp0 + c0 - 1
                if val > dp2: dp2 = val
                # transitions for f0
                val = c0 + deg_u - 1
                if val > dp0: dp0 = val
                # update ret for next child
                if c1 > ret: ret = c1
                if c2 > ret: ret = c2
            f0[u], f1[u], f2[u], f3[u] = dp0, dp1, dp2, dp3
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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]