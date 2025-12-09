import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class TreeColoring_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3177
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices, labeled from `0` to `{N_minus_1}`.

The tree contains the following {N} - 1 = {N_minus_1} undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning there is an undirected edge **connecting vertex `u` to vertex `v` with weight `w`:
{edges}

Your task is to **select exactly {K} distinct vertices**. These selected vertices are called **colored**, and the remaining {N} - {K} = {N_minus_K} vertices are called **uncolored**. Try your best to **maximize the total distance**, defined as:
- The sum of all pairwise distances **between colored vertices**,
- Plus the sum of all pairwise distances **between uncolored vertices**.

(Note: Since the graph is a tree, there is exactly one unique path between any two vertices.)

**Output Format:**
Your final answer should be a single line containing the {K} selected (colored) vertices in any order, separated by **spaces**.
Example: `{first_K_vertices}` (do **NOT** include the backticks or quotes)."""
    
    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 2.0,
                 **kwargs) :
        """
        Initialize the TreeColoring_Environment instance.
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

        K = self.parameter["K"] = random.randint(1, N - 1)

        edges = self.parameter["edges"] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v, random.randint(1, N)))
        random.shuffle(edges)
        
        for u, v, w in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, w in edges)) == N - 1

        tree = networkx.Graph()
        tree.add_weighted_edges_from(edges)
        assert networkx.is_tree(tree)


        adjacency_list = [[] for s in range(N)]
        for u, v, w in edges :
            adjacency_list[u].append((v, w))
            adjacency_list[v].append((u, w))

        dpF = [[None] * (K + 1) for u in range(N)]
        decisions = [[] for u in range(N)]
        Size = [0] * N
        def DP(u, parent) :
            Size[u] = 1
            dpF[u][0] = 0
            if K :
                dpF[u][1] = 0
            for v, w in adjacency_list[u] :
                if v == parent :
                    continue
                DP(v, u)
                decision = decisions[u]
                decision.append((v, w, [None] * (min(Size[u] + Size[v], K) + 1)))
                decision = decision[-1][-1]
                for uk in range(min(Size[u], K), -1, -1) :
                    for vk in range(min(Size[v], K - uk), -1, -1) :
                        assert uk + vk <= K
                        if dpF[u][uk] is None or dpF[v][vk] is None :
                            continue
                        if (N - K) < (Size[v] - vk) :
                            continue
                        val = dpF[u][uk] + dpF[v][vk] + w * (vk * (K - vk) + (Size[v] - vk) * ((N - K) - (Size[v] - vk)))
                        if dpF[u][uk + vk] is None or dpF[u][uk + vk] <= val :
                            dpF[u][uk + vk] = val
                            decision[uk + vk] = vk
                Size[u] += Size[v]
        DP(0, -1)
        assert dpF[0][K]
        self.parameter["reference_answer_distance"] = dpF[0][K]

        self.parameter["reference_answer"] = []
        def DFS(u, k) :
            if Size[u] == 1 :
                assert len(decisions[u]) == 0
            decisions[u].reverse()
            for decision in decisions[u] :
                v, vk = decision[0], decision[-1][k]
                k -= vk
                DFS(v, vk)
            assert k in (0, 1)
            if k == 1 :
                self.parameter["reference_answer"].append(u)
        DFS(0, K)
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["reference_answer"]))
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        K = self.parameter["K"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            K = K,
            N_minus_K = N - K,
            first_K_vertices = " ".join(map(str, range(K))),
            edges = "\n".join("({}, {}, {})".format(u, v, w) for u, v, w in self.parameter["edges"]),
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

            colored_vertices = processed_result
            if len(colored_vertices) != self.parameter["K"] :
                return self.rewards["invalid_solution"]
            if len(set(colored_vertices)) != self.parameter["K"] :
                return self.rewards["invalid_solution"]
            if not all((0 <= vertex < self.parameter["N"]) for vertex in colored_vertices) :
                return self.rewards["invalid_solution"]
            
            adjacency_list = [[] for s in range(self.parameter["N"])]
            for u, v, w in self.parameter["edges"] :
                adjacency_list[u].append((v, w))
                adjacency_list[v].append((u, w))

            colored = [0] * self.parameter["N"]
            for colored_vertex in colored_vertices :
                colored[colored_vertex] = 1
            Size = [0] * self.parameter["N"]
            answer = 0
            def DFS(u, parent) :
                nonlocal answer
                Size[u] = 1
                for v, w in adjacency_list[u] :
                    if v == parent :
                        continue
                    DFS(v, u)
                    answer += w * (colored[v] * (self.parameter["K"] - colored[v]) + (Size[v] - colored[v]) * ((self.parameter["N"] - self.parameter["K"]) - (Size[v] - colored[v])))
                    Size[u] += Size[v]
                    colored[u] += colored[v]
            DFS(0, -1)
            gold = self.parameter["reference_answer_distance"]
            assert answer <= gold

            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise ValueError("Invalid rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]