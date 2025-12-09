import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class TriumphalArch_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3554
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from `0` to `{N_minus_1}`. The edges of the tree are given as follows:
{edges}

Alice and Bob are playing a game on this tree:
- Initially, Bob is standing at vertex `0`. The vertex `0` is already marked as **(permanently) black**, and all other vertices are **white**.
- On each turn:
  - Alice first chooses any K vertices and marks them as "(permanently) black".
  - Then, Bob may move to any vertex adjacent to his current position.
- If Bob ever reaches a **non-black** vertex on any turn, he wins. If eventually **all vertices become black**, then Alice wins.

Assuming both players play optimally, what is the **minimum value of K** such that Alice is guaranteed to win?"""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the TriumphalArch_Environment instance.
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
        assert N >= 3, "N should be greater than or equal to 3"

        edges = self.parameter["edges"] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v))
        random.shuffle(edges)
        
        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)) == N - 1


        # Build adjacency list (0-indexed)
        S = [[] for _ in range(N)]
        for u, v in edges:
            S[u].append(v)
            S[v].append(u)

        # son[u] = number of children of u in the rooted tree at 0
        son = [0] * N
        def dfs1(u, p):
            for v in S[u]:
                if v == p:
                    continue
                son[u] += 1
                dfs1(v, u)

        dfs1(0, -1)

        # Binary search bounds: k in [L..R]
        L = son[0]
        R = max(son)

        # f[u] will hold the DP value for subtree rooted at u
        f = [0] * N
        def dfs2(u, p, k):
            # Start with son[u] - k
            total = son[u] - k
            for v in S[u]:
                if v == p:
                    continue
                dfs2(v, u, k)
                if f[v] > 0:
                    total += f[v]
            f[u] = total

        ans = R
        while L <= R:
            mid = (L + R) // 2
            dfs2(0, -1, mid)
            # If f[0] <= 0, A can win with k = mid
            if f[0] <= 0:
                ans = mid
                R = mid - 1
            else:
                L = mid + 1

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