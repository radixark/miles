import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class GasFireExtinguishers_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3479
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from `0` to `{N_minus_1}`. The tree has the following {N_minus_1} undirected edges. Each edge is represented as a tuple `(u, v)`, meaning there is an undirected edge connecting vertex `u` and vertex `v`:
{edges}

There is an array C[0], C[1], ..., C[{N_minus_1}], all initially set to 0. For each vertex `u` (0 ≤ u < {N}), you must choose a vertex `P[u]` such that the distance (in number of edges) from `u` to `P[u]` is at most {K}; then, increment C[P[u]] by 1.
Try your best to **minimize** the total value of ceil(C[0] / {S}) + ceil(C[1] / {S}) + ... + ceil(C[{N_minus_1}] / {S}), where `ceil(x)` means rounding `x` up to the nearest integer. Output a single line containing `P[0]`, `P[1]`, ..., `P[{N_minus_1}]`, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the GasFireExtinguishers_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
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

        G = networkx.Graph()
        G.add_edges_from(edges)
        distances = dict(networkx.all_pairs_shortest_path_length(G))
        K = self.parameter["K"] = random.randint(1, max(1, max(distances[u][v] for u in range(N) for v in range(N)) // 2))
        self.parameter["valid_P"] = [[v for v in range(N) if distances[u][v] <= K] for u in range(N)]
        S = self.parameter["S"] = random.randint(2, max(2, N // K))


        # Build adjacency list for 0-indexed rooms
        graph = [[] for _ in range(N)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        # f[u][i]: number of rooms in subtree u at distance exactly i that still need an extinguisher
        # g[u][i]: capacity of extinguishers at u that can serve rooms at distance exactly i
        f = [[0] * (K + 1) for _ in range(N)]
        g = [[0] * (K + 1) for _ in range(N)]
        ans = 0

        def dfs(u, parent):
            nonlocal ans
            f[u][0] = 1
            # accumulate from children
            for v in graph[u]:
                if v == parent:
                    continue
                dfs(v, u)
                for i in range(K):
                    f[u][i + 1] += f[v][i]
                    g[u][i + 1] += g[v][i]
            # place new extinguishers for rooms at distance K in subtree
            need = (f[u][K] + S - 1) // S
            ans += need
            # capacity left in newly placed extinguishers
            l = need * S - f[u][K]
            f[u][K] = 0
            g[u][0] += l
            # match needs and capacities within K
            # first for exact K distance pairs
            for i in range(K + 1):
                j = K - i
                d = min(f[u][i], g[u][j])
                f[u][i] -= d
                g[u][j] -= d
            # then for distance K-1 pairs
            for i in range(K):
                j = K - 1 - i
                d = min(f[u][i], g[u][j])
                f[u][i] -= d
                g[u][j] -= d

        # run DFS from root 0
        dfs(0, -1)

        # final matching at root
        for i in range(K + 1):
            for j in range(K + 1):
                if i + j <= K:
                    d = min(f[0][i], g[0][j])
                    f[0][i] -= d
                    g[0][j] -= d
        # remaining rooms need extinguishers
        tot = sum(f[0][i] for i in range(K + 1))
        ans += (tot + S - 1) // S

        assert ans > 0, "The answer should be greater than 0"
        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("{} {}".format(u, v) for u, v in self.parameter["edges"]),
            K = self.parameter["K"],
            S = self.parameter["S"],
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

            if len(processed_result) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            
            C = [0] * self.parameter["N"]
            for u, P_u in enumerate(processed_result) :
                if P_u not in self.parameter["valid_P"][u] :
                    return self.rewards["invalid_solution"]
                C[P_u] += 1
            
            answer, gold = sum((C[u] + self.parameter["S"] - 1) // self.parameter["S"] for u in range(self.parameter["N"])), self.parameter["gold_answer"]
            assert 0 < gold <= answer, "gold should be greater than 0 and less than or equal to answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise ValueError("Invalid rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]