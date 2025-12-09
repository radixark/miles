import random
import networkx
from collections import deque
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinimumSteinerTree_Environment(VerifiableEnvironment) : # Submitted to https://www.luogu.com.cn/problem/P6192
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning an undirected edge **connecting vertex `u` to vertex `v` with weight `w`**:
{edges}

Your task is to select a subset of edges `T = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)]` such that:
- The selected edges form a **connected graph** that contains these {K} verticies: {to_be_connected}
- Your goal is to **minimize** the total weight of the selected edges: `w_1 + w_2 + ... + w_k`.

**Output Format:**
Your final answer should be a single line containing the endpoints of the selected edges in order: `u_1 v_1 u_2 v_2 ... u_k v_k`, separated by **spaces**. Example: `0 1 1 2 2 3` (do **NOT** include the backticks or quotes); this means the it includes the edges `(0, 1, w_1)`, `(1, 2, w_2)`, and `(2, 3, w_3)`"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinimumSteinerTree_Environment instance.
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
        assert N >= 4, "N should be greater than or equal to 4"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        edges = self.parameter["edges"] = []

        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v, random.randint(1, N)))
        
        num_edges = int(edge_density * N * (N - 1) / 2)
        if len(edges) < num_edges :
            remaining_edges = list(set((u, v) for u in range(N) for v in range(u + 1, N)) - set((u, v) for u, v, w in edges))
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for u, v in remaining_edges :
                edges.append((u, v, random.randint(1, N)))
        random.shuffle(edges)

        for u, v, w in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, w in edges)), "edges should be unique"

        K = self.parameter["K"] = random.randint(3, min(20, N - 1))

        to_be_connected = self.parameter["to_be_connected"] = random.sample(range(N), K)


        adj = [[] for _ in range(N)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        full_mask = (1 << K) - 1
        dp = [[None] * (full_mask + 1) for _ in range(N)]
        for i in range(K):
            dp[to_be_connected[i]][1 << i] = 0
        for s1 in range(1, full_mask + 1):
            for i in range(N):
                s2 = (s1 - 1) & s1
                while s2:
                    a = dp[i][s2]
                    b = dp[i][s1 ^ s2]
                    if a is not None and b is not None:
                        v = a + b
                        cur = dp[i][s1]
                        if cur is None or v < cur:
                            dp[i][s1] = v
                    s2 = (s2 - 1) & s1
            vis = [False] * N
            q = deque()
            for i in range(N):
                if dp[i][s1] is not None:
                    q.append(i)
                    vis[i] = True
            while q:
                u = q.popleft()
                vis[u] = False
                du = dp[u][s1]
                for v, w in adj[u]:
                    nd = du + w
                    cur = dp[v][s1]
                    if cur is None or nd < cur:
                        dp[v][s1] = nd
                        if not vis[v]:
                            q.append(v)
                            vis[v] = True
        self.parameter["gold_answer"] = dp[to_be_connected[0]][full_mask]
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {}, {})".format(u, v, w) for u, v, w in self.parameter["edges"]),
            K = self.parameter["K"],
            to_be_connected = " ".join(map(str, self.parameter["to_be_connected"])),
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

            mst = processed_result
            if len(mst) % 2 != 0 :
                return self.rewards["wrong_format"]
            mst = [(mst[i], mst[i + 1]) for i in range(0, len(mst), 2)]
            
            if not (set(range(self.parameter["N"])) >= (set(u for u, v in mst) | set(v for u, v in mst)) >= set(self.parameter["to_be_connected"])) :
                return self.rewards["invalid_solution"]

            subgraph = networkx.Graph()
            edge2weight = {(u, v) : w for u, v, w in self.parameter["edges"]}            
            answer_weight = 0
            for u, v in mst :
                u, v = min(u, v), max(u, v)
                if (u, v) not in edge2weight :
                    return self.rewards["invalid_solution"]
                answer_weight += edge2weight[(u, v)]
                subgraph.add_edge(u, v)
            if not networkx.is_connected(subgraph) :
                return self.rewards["invalid_solution"]
            
            assert self.parameter["gold_answer"] <= answer_weight, "answer_weight should be greater than or equal to gold_answer"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((self.parameter["gold_answer"] / answer_weight) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == answer_weight)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]