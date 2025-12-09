import random
import networkx
from typing import Optional
from collections import deque
from Gym.environment import VerifiableEnvironment


class MaxTree_KPathCoverahe_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4551
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from `0` to `{N_minus_1}`. The tree contains the following {N_minus_1} undirected edges. Each edge is represented as a tuple `(u, v)`, meaning there is an undirected edge **connecting vertex `u` and vertex `v`**:
{edges}

You need to choose exactly {K} unordered pairs of distinct vertices `(u, v)`. For each selected pair, define the set of all vertices on the unique path between `u` and `v` (inclusive) as `covered`. Please **maximize the total number of unique vertices that are covered by at least one of the {K} paths**. Output a single integer — the maximum number of vertices that can be covered."""
    
    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MaxTree_KPathCoverahe_Environment instance.
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
        degrees = [0] * N
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v))
            degrees[u] += 1
            degrees[v] += 1
        random.shuffle(edges)

        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)) == N - 1

        tree = networkx.Graph()
        tree.add_edges_from(edges)
        assert networkx.is_tree(tree)

        K = self.parameter["K"] = random.randint(1, max(1, sum(degree == 1 for degree in degrees) // 2 - 1))


        M = K * 2

        # Build adjacency list (0-indexed)
        adjacency = [[] for _ in range(N)]
        for A, B in edges:
            adjacency[A].append(B)
            adjacency[B].append(A)

        # d[i] = number of remaining neighbors of i before it becomes a "leaf" in peeling
        d = [len(adjacency[i]) - 1 for i in range(N)]

        # dep[i] = round at which node i is peeled (distance from nearest original leaf, plus one)
        dep = [0] * N
        q = deque()

        # Initialize queue with all initial leaves (d[i] == 0)
        for i in range(N):
            if d[i] == 0:
                q.append(i)
                dep[i] = 1

        # cnt[k] = number of nodes peeled at round k
        cnt = [0] * (N + 1)
        maxd = 0

        # Perform the "topological peeling" of the tree
        while q:
            x = q.popleft()
            depth = dep[x]
            cnt[depth] += 1
            if depth > maxd:
                maxd = depth
            for y in adjacency[x]:
                d[y] -= 1
                if d[y] == 0:
                    dep[y] = depth + 1
                    q.append(y)

        # Sum, for each layer, the minimum of its size or M = 2 * L
        ans = 0
        for k in range(1, maxd + 1):
            ans += min(cnt[k], M)
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            K = self.parameter["K"],
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