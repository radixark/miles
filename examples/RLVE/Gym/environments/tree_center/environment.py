import random
import networkx
from typing import Optional
from Gym.environment import VerifiableEnvironment


class TreeCenter_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices, labeled from `0` to `{N_minus_1}`.

Each vertex has a cost, given as a list `C` of length {N}, where `C[i]` is the cost of vertex i:
{C}

The tree contains the following {N} - 1 = {N_minus_1} undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning there is an undirected edge **connecting vertex `u` to vertex `v` with weight `w`:
{edges}

Your task is to select a single vertex `r` (where `r` is in the range 0 to {N_minus_1}).
Try your best to **minimize** dist(0, r) * C[0] + dist(1, r) * C[1] + ... + dist({N_minus_1}, r) * C[{N_minus_1}], where `dist(i, j)` is the distance between vertices i and j in the tree. The distance between two vertices is defined as the sum of the weights of the edges on the unique path connecting them (since the graph is a tree, there is exactly one unique path between any two vertices).

**Output Format:** Your final answer should be a single integer `r` (the index of the selected vertex). Example: `0` (do **NOT** include the backticks or quotes)."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 8.0,
                 **kwargs) :
        """
        Initialize the TreeCenter_Environment instance.
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
        assert N >= 2, "N should be greater than or equal to 2"

        C = self.parameter["C"] = [random.randint(1, N) for vertex in range(N)]

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


        adjacent = [[] for u in range(N)]
        for u, v, w in edges :
            adjacent[u].append((v, w))
            adjacent[v].append((u, w))
        
        self.parameter["reference_answer"] = 0
        self.parameter["gold_answer"] = 0
        subtree_sumC = [0] * N
        def DFS(u : int, parent : int, depth : int) -> None :
            subtree_sumC[u] = C[u]
            self.parameter["gold_answer"] += depth * C[u]
            for v, w in adjacent[u] :
                if v == parent :
                    continue
                DFS(v, u, depth + w)
                subtree_sumC[u] += subtree_sumC[v]
        DFS(0, -1, 0)

        def FindSolution(u : int, parent : int, now_answer : int) :
            if now_answer < self.parameter["gold_answer"] :
                self.parameter["reference_answer"] = u
                self.parameter["gold_answer"] = now_answer
            for v, w in adjacent[u] :
                if v == parent :
                    continue
                FindSolution(v, u, now_answer + (subtree_sumC[0] - subtree_sumC[v]) * w - subtree_sumC[v] * w)
        FindSolution(0, -1, self.parameter["gold_answer"])
        assert self.parameter["gold_answer"] > 0
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            C = "\n".join("C[{}]={}".format(i, Ci) for i, Ci in enumerate(self.parameter["C"])),
            edges = "\n".join("({}, {}, {})".format(u, v, w) for u, v, w in self.parameter["edges"]),
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
            root = processed_result
            if not (0 <= root < self.parameter["N"]) :
                return self.rewards["wrong_format"]

            adjacent = [[] for u in range(self.parameter["N"])]
            for u, v, w in self.parameter["edges"] :
                adjacent[u].append((v, w))
                adjacent[v].append((u, w))
            
            gold, answer = self.parameter["gold_answer"], 0
            def DFS(u : int, parent : int, depth : int) -> None :
                nonlocal answer
                answer += depth * self.parameter["C"][u]
                for v, w in adjacent[u] :
                    if v == parent :
                        continue
                    DFS(v, u, depth + w)
            DFS(root, -1, 0)


            assert gold <= answer, "gold <= answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]