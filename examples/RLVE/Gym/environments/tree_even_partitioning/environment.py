import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class TreeEvenPartitioning_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3915
    prompt_template = \
r"""You have a **tree** (i.e., a connected undirected graph with no cycles) with {NK} vertices labeled from `1` to `{NK}`. The tree contains the following {NK} - 1 undirected edges. Each edge is represented as a tuple `(u, v)`, meaning there is an undirected edge connecting vertex `u` to vertex `v`:
{edges}

Partition all vertices into {N} **disjoint** sets such that: (1) each set contains exactly {K} vertices ({K} = {NK} / {N}), AND (2) each set forms a connected subgraph of the tree. Output {N} lines - each line should contain the {K} vertices of one set, separated by spaces; the vertices within a set and the sets themselves may be in any order."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(connected/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the TreeEvenPartitioning_Environment instance.
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
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 2, "MAX_N should be greater than or equal to 2"

        assert "MAX_K" in self.parameter, "MAX_K is required in parameter"
        MAX_K = self.parameter["MAX_K"]
        assert MAX_K >= 2, "MAX_K should be greater than or equal to 2"

        N = self.parameter["N"] = random.randint(2, MAX_N)
        K = self.parameter["K"] = random.randint(2, MAX_K)

        groups = list(range(1, N * K + 1))
        random.shuffle(groups)
        groups = [groups[i * K : (i + 1) * K] for i in range(N)]

        edges = self.parameter["edges"] = []

        for i, group in enumerate(groups) :
            assert len(group) == K, f"Group {i} should have exactly {K} vertices"
            for index, vertex in enumerate(group) :
                if index == 0 :
                    continue
                u, v = vertex, group[random.randint(0, index - 1)]
                u, v = min(u, v), max(u, v)
                edges.append((u, v))
            if i == 0 :
                continue
            u, v = random.choice(group), random.choice(groups[random.randint(0, i - 1)])
            u, v = min(u, v), max(u, v)
            edges.append((u, v))
        
        random.shuffle(edges)
        
        for u, v in edges :
            assert 1 <= u < v <= N * K
        assert len(edges) == len(set(edges)) == N * K - 1

        tree = networkx.Graph()
        tree.add_edges_from(edges)
        assert networkx.is_tree(tree)

        self.parameter["reference_answer"] = "\n".join(" ".join(map(str, group)) for group in groups)
    

    def _prompt_generate(self) -> str :
        N, K = self.parameter["N"], self.parameter["K"]
        return self.prompt_template.format(
            NK = N * K,
            N = N,
            K = K,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[List[List[int]]] :
        if answer is not None :
            answer = answer.strip()
            try :
                groups = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        groups.append(list(map(int, line.split())))
                        if len(groups[-1]) != self.parameter["K"] :
                            return None
                if len(groups) != self.parameter["N"] :
                    return None
                return groups
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if set(vertex for group in processed_result for vertex in group) != set(range(1, self.parameter["N"] * self.parameter["K"] + 1)) :
                return self.rewards["invalid_solution"]
            
            labels = [None] * (self.parameter["N"] * self.parameter["K"] + 1)
            for label, group in enumerate(processed_result) :
                assert 0 <= label < self.parameter["N"], f"Label {label} is out of range"
                assert len(group) == self.parameter["K"], f"Group {group} should have exactly {self.parameter['K']} vertices"
                for vertex in group :
                    assert labels[vertex] is None, f"Vertex {vertex} is already labeled"
                    labels[vertex] = label
            edge_numbers = [0] * self.parameter["N"]
            for u, v in self.parameter["edges"] :
                if labels[u] == labels[v] :
                    edge_numbers[labels[u]] += 1
            
            assert all(0 <= edge_number <= self.parameter["K"] - 1 for edge_number in edge_numbers), "Edge numbers are out of range"
            connected = sum(int(edge_number == self.parameter["K"] - 1) for edge_number in edge_numbers)
            assert connected <= self.parameter["N"], "Connected components exceed N"
            if self.rewards["rewarding_strategy"] == "(connected/all)^beta" :
                return self.rewards["rewarding_weight"] * ((connected / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (connected == self.parameter["N"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]