
import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MultiDrink_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3549
    prompt_template = \
r"""There is a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from `0` to `{N_minus_1}`. Its edges are:
{edges}

Please find a permutation of the vertices p[0], p[1], ..., p[{N_minus_1}] such that for every pair (p[i], p[i + 1]) with 0 ≤ i < {N_minus_1}, the distance between p[i] and p[i + 1] in the tree (measured in number of edges) is **at most 2**. Output the permutation as a single line of space-separated integers in order."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MultiDrinkProblem.
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
        assert "N" in self.parameter, "N must be specified in parameters"
        N = self.parameter["N"]
        assert N >= 4, "N must be at least 4"

        edges = self.parameter["edges"] = []
        neighbors = [[] for _ in range(N)]
        def add_edge(u, v) :
            edges.append((min(u, v), max(u, v)))
            neighbors[u].append(v)
            neighbors[v].append(u)
        
        paths = [[u] for u in range(N)]
        while len(paths) > 1 :
            while True :
                i, j = random.choices(range(len(paths)), k = 2, weights = [len(path) for path in paths])
                if i != j :
                    break
            path_i, path_j = paths[i], paths[j]
            
            a, b = path_i[-1], path_j[0]
            if random.random() < 0.5 :
                add_edge(a, random.choice([b] + neighbors[b]))
            else :
                add_edge(b, random.choice([a] + neighbors[a]))

            paths = [path for index, path in enumerate(paths) if index not in (i, j)] + [path_i + path_j]
        self.parameter["reference_answer"] = " ".join(map(str, paths[0]))

        random.shuffle(edges)

        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"

        tree = networkx.Graph()
        tree.add_edges_from(edges)
        assert networkx.is_tree(tree)
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
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

            P = processed_result
            if len(P) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if set(P) != set(range(self.parameter["N"])) :
                return self.rewards["invalid_solution"]
            
            neighbors = [set() for _ in range(self.parameter["N"])]
            for u, v in self.parameter["edges"] :
                neighbors[u].add(v)
                neighbors[v].add(u)
            
            satisfied = sum(int((a in neighbors[b]) or (len(neighbors[a] & neighbors[b]) > 0)) for a, b in zip(P, P[1 :]))
            assert satisfied <= self.parameter["N"] - 1, "satisfied should be at most N - 1"
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / (self.parameter["N"] - 1)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == self.parameter["N"] - 1)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]