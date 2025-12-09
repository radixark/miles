import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class TreeRandomWalkExpectation_Environment(VerifiableEnvironment) : # https://www.luogu.com.cn/problem/P3412
    prompt_template = \
r"""You are given a **tree** with {N} vertices labeled from 0 to {N_minus_1}. The tree has the following {N_minus_1} undirected edges:
{edges}

A random walk on the tree is defined as follows: from the current vertex, you move to one of its neighbors uniformly at random at each step. Define E(S, T) as the expected number of steps to reach vertex T starting from vertex S (the walk stops immediately upon reaching T).

Please compute the sum of all E(S, T) over all ordered pairs (S, T), divided by {N}². Output this value modulo {MOD}.

**Output Format:** A single integer — the value of (∑ E(S, T)) / {N}² modulo {MOD}."""
    MOD = 998244353

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the TreeRandomWalkExpectation_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 2"

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


        adj = [[] for _ in range(N)]
        d = [0] * N

        # Read edges, build adjacency and initial degree array
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            d[u] += 1
            d[v] += 1

        totd = sum(d)

        sz = [0] * N
        parent = [-1] * N

        # DFS to compute subtree sizes and accumulate degree-sums
        def dfs(u, p):
            parent[u] = p
            sz[u] = 1
            for v in adj[u]:
                if v == p:
                    continue
                dfs(v, u)
                sz[u] += sz[v]
                d[u] += d[v]

        dfs(0, -1)

        # modular inverse of n^2
        rev = pow(N * N % self.MOD, self.MOD - 2, self.MOD)

        ans = 0
        for u in range(N):
            for v in adj[u]:
                if v == parent[u]:
                    # edge from u up to its parent
                    ans = (ans + d[u] * sz[u] * (N - sz[u])) % self.MOD
                else:
                    # edge from u down to child v
                    ans = (ans + (totd - d[v]) * sz[v] * (N - sz[v])) % self.MOD

        self.parameter["reference_answer"] = ans * rev % self.MOD
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
            MOD = self.MOD,
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
            if not (0 <= processed_result < self.MOD) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]