import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class ImpParty_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3524
    prompt_template = \
r"""You are given an **undirected graph** with 3 × {N} vertices, labeled from `0` to `{ThreeN_minus_1}`. The graph contains the following undirected edges:
{edges}

It is guaranteed that the graph contains a **clique of size 2 × {N}** — a set of 2 × {N} vertices in which every pair is connected by an edge.
Your task is to find any **clique of size {N}** in the graph. Output the indices of the selected {N} vertices, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the ImpParty_Environment instance.
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

        constructed_clique = random.sample(range(3 * N), 2 * N)
        for u in constructed_clique :
            for v in constructed_clique :
                if u < v :
                    edges.append((u, v))
        
        not_in_constructed_clique = list(set(range(3 * N)) - set(constructed_clique))
        edges += random.sample([(min(u, v), max(u, v)) for u in constructed_clique for v in not_in_constructed_clique], random.randint(0, len(constructed_clique) * len(not_in_constructed_clique)))

        random.shuffle(edges)

        for u, v in edges :
            assert 0 <= u < v < 3 * N, "edges should be within the range of 0 to 3N-1"
        assert len(edges) == len(set(edges)), "edges should be unique"

        self.parameter["reference_answer"] = " ".join(map(str, random.sample(constructed_clique, N)))
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            ThreeN_minus_1 = 3 * N - 1,
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

            clique = processed_result
            if len(clique) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= vertex < 3 * self.parameter["N"] for vertex in clique) :
                return self.rewards["invalid_solution"]
            if len(set(clique)) != len(clique) :
                return self.rewards["invalid_solution"]
            
            satisfied = 0
            edges = set(map(tuple, self.parameter["edges"]))
            for u in clique :
                for v in clique :
                    if u < v :
                        satisfied += int((u, v) in edges)
            assert satisfied <= self.parameter["N"] * (self.parameter["N"] - 1) // 2, "satisfied edges should not exceed N choose 2"
            
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / (self.parameter["N"] * (self.parameter["N"] - 1) // 2)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == (self.parameter["N"] * (self.parameter["N"] - 1) // 2))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]