import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class POLPolarization_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3563
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from `0` to `{N_minus_1}`. The tree contains the following {N_minus_1} undirected edges. Each edge is represented as a tuple `(u, v)`, meaning there is an undirected edge **connecting vertex `u` and vertex `v`**:
{edges}

Your task is to assign a direction to each edge (i.e., for each edge `(u, v)`, you may direct it either from `u` to `v` or from `v` to `u`) to form a **directed tree**. Try your best to **maximize** the number of ordered pairs `(X, Y)` such that `X ≠ Y` and vertex `X` can **reach** vertex `Y` along directed edges (i.e., `Y` is reachable from `X` in the directed tree). Output a single integer — the maximum number of such ordered pairs `(X, Y)`."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the POLPolarization_Environment instance.
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
            u, v = min(u, v), max(u, v)
            edges.append((u, v))
        random.shuffle(edges)

        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)) == N - 1


        adjacency = [[] for _ in range(N)]
        for u, v in edges:
            adjacency[u].append(v)
            adjacency[v].append(u)

        # First DFS: compute subtree sizes and "max part" sizes to find the centroid
        siz = [0] * N
        msiz = [0] * N
        rt = 0
        best_ms = N

        def dfs(p, fa):
            nonlocal rt, best_ms
            siz[p] = 1
            max_sub = 0
            for v in adjacency[p]:
                if v == fa:
                    continue
                dfs(v, p)
                siz[p] += siz[v]
                if siz[v] > max_sub:
                    max_sub = siz[v]
            # consider the "upward" part when p is removed
            up = N - siz[p]
            if up > max_sub:
                max_sub = up
            msiz[p] = max_sub
            # update centroid if this node is better
            if max_sub < best_ms:
                best_ms = max_sub
                rt = p

        dfs(0, -1)

        # Second DFS from centroid: recompute subtree sizes and record parents
        siz = [0] * N
        parent = [-1] * N

        def dfs2(p, fa):
            siz[p] = 1
            parent[p] = fa
            for v in adjacency[p]:
                if v == fa:
                    continue
                dfs2(v, p)
                siz[p] += siz[v]

        dfs2(rt, -1)

        # initial answer: sum of sizes of all subtrees except the centroid itself
        ans = sum(siz[i] for i in range(N) if i != rt)

        # count how many child-subtrees of each size the centroid has
        cnt = [0] * (N + 1)
        for v in adjacency[rt]:
            if parent[v] == rt:
                cnt[siz[v]] += 1

        # merge pairs of equal sizes greedily
        for i in range(1, N // 2 + 1):
            while cnt[i] > 2:
                cnt[i] -= 2
                cnt[2 * i] += 1

        # subset‐sum via bitset in an integer
        dp = 1
        for i in range(1, N + 1):
            for _ in range(cnt[i]):
                dp |= dp << i

        # find the best split i ≤ N//2 that is reachable
        half = N // 2
        for i in range(half, -1, -1):
            if (dp >> i) & 1:
                ans += i * (N - i - 1)
                break

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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]