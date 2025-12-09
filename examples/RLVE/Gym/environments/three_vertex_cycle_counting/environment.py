import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class ThreeVertexCycleCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1989
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.
The graph contains the following undirected edges:
{edges}

Please count the number of distinct **three‐vertex cycles** in the graph (the order of vertices in the cycle does not matter, and cycles are considered distinct if they have different sets of vertices)."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the ThreeVertexCycleCounting_Environment instance.
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

        assert "edge_ratio" in self.parameter, "edge_ratio is required in parameter"
        edge_ratio = self.parameter["edge_ratio"]

        edges = self.parameter["edges"] = random.sample([(u, v) for u in range(N) for v in range(u + 1, N)], max(1, min(N * (N - 1) // 2, int(edge_ratio * N))))
        random.shuffle(edges)
        
        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"


        degree = [0] * N
        for u, v in edges :
            degree[u] += 1
            degree[v] += 1

        # build adjacency lists with edges directed from lower‐degree to higher‐degree endpoint
        adj = [[] for _ in range(N)]
        for u, v in edges:
            a, b = u, v
            if degree[a] > degree[b] or (degree[a] == degree[b] and a > b):
                a, b = b, a
            adj[a].append(b)

        # count triangles
        vis = [False] * N
        ans = 0
        for i in range(N):
            # mark all neighbors of i
            for j in adj[i]:
                vis[j] = True
            # for each two‐hop path i→j→k, check if k is also a neighbor of i
            for j in adj[i]:
                for k in adj[j]:
                    if vis[k]:
                        ans += 1
            # unmark
            for j in adj[i]:
                vis[j] = False

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
                if processed_result == 0 :
                    return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == 0)
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]