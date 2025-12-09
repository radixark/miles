import random
import networkx
from typing import Optional, Tuple
from Gym.environment import VerifiableEnvironment


class MaxTreeXorPath_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4551
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices, labeled from `0` to `{N_minus_1}`. The tree contains the following {N} - 1 = {N_minus_1} undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning there is an undirected edge **connecting vertex `u` to vertex `v` with weight `w`**:
{edges}

Please find a pair of vertices (`u`, `v`) to **maximize the bitwise XOR of all edge weights on the unique path** connecting them.

**Output Format:** Your final answer should be two integers `u` and `v` (the indices of the selected vertices), separated by a space. Example: `0 1` (do **NOT** include the backticks or quotes)."""

    def __init__(self,
                 lower_max_weight : int = 2 ** 4,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MaxTreeXorPath_Environment instance.
        """
        super().__init__(**kwargs)

        self.lower_max_weight = lower_max_weight

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        max_weight = self.lower_max_weight
        while max_weight <= N * 2 :
            max_weight *= 2

        edges = self.parameter["edges"] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v, random.randint(1, max_weight - 2)))
        random.shuffle(edges)
        
        for u, v, w in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, w in edges)) == N - 1

        tree = networkx.Graph()
        tree.add_weighted_edges_from(edges)
        assert networkx.is_tree(tree)


        adj = [[] for _ in range(N)]
        for u, v, w in edges :
            adj[u].append((v, w))
            adj[v].append((u, w))

        Xor = self.parameter["Xor"] = [0] * N
        def dfs(u, parent):
            for v, w in adj[u]:
                if v == parent:
                    continue
                Xor[v] = Xor[u] ^ w
                dfs(v, u)
        dfs(0, -1)

        Ans_u, Ans_v = 0, 1
        for u in range(N) :
            for v in range(u + 1, N) :
                if (Xor[u] ^ Xor[v]) > (Xor[Ans_u] ^ Xor[Ans_v]) :
                    Ans_u, Ans_v = u, v
        
        self.parameter["reference_answer"] = "{} {}".format(Ans_u, Ans_v)
        self.parameter["gold_answer"] = (Xor[Ans_u] ^ Xor[Ans_v])
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {}, {})".format(u, v, w) for u, v, w in self.parameter["edges"]),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[Tuple[int, int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                u, v = map(int, answer.split())
                return u, v
            except :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            u, v = processed_result
            if not (0 <= u < self.parameter["N"] and 0 <= v < self.parameter["N"]) :
                return self.rewards["wrong_format"]
            
            answer, gold = (self.parameter["Xor"][u] ^ self.parameter["Xor"][v]), self.parameter["gold_answer"]
            assert answer <= gold, "answer <= gold"

            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]