import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class EvenDegreeGraphPartitioning_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3429
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from 0 to {N_minus_1}. The graph contains the following undirected edges:
{edges}

Please partition the vertices into two groups (labeled 1 and 2) such that:
1. Each vertex belongs to exactly one group.
2. For each vertex, the number of edges connecting it to vertices in the **same** group is even.

**Output Format:** A single line containing {N} integers (separated by spaces), where the i-th integer is the group number (1 or 2) assigned to vertex i (from 0 to {N_minus_1})."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
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

        while True :
            vertex_permutation = list(range(N))
            random.shuffle(vertex_permutation)
            group_1 = vertex_permutation[: random.randint(0, N)]
            group_2 = vertex_permutation[len(group_1) :]
            
            edges = self.parameter["edges"] = []

            degrees = [0] * N
            def build(group) :
                if len(group) <= 2 :
                    return
                for i in range(1, len(group) - 1) :
                    neighbors = random.sample(group[: i], random.randint(0, i))
                    for neighbor in neighbors :
                        u, v = min(group[i], neighbor), max(group[i], neighbor)
                        edges.append((u, v))
                        degrees[u] += 1
                        degrees[v] += 1
                for vertex in group[: -1] :
                    if degrees[vertex] % 2 == 1 :
                        u, v = min(group[-1], vertex), max(group[-1], vertex)
                        edges.append((u, v))
                        degrees[u] += 1
                        degrees[v] += 1
                assert all(degrees[vertex] % 2 == 0 for vertex in group), "All vertices in the group should have even degree"
            build(group_1)
            build(group_2)

            if len(group_1) and len(group_2) :
                edges += random.sample([(min(u, v), max(u, v)) for u in group_1 for v in group_2], random.randint(0, len(group_1) * len(group_2)))

            if len(edges) > 0 :
                break

        random.shuffle(edges)

        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"

        labels = [0] * N
        for i in range(len(group_1)) :
            labels[group_1[i]] = 1
        for i in range(len(group_2)) :
            labels[group_2[i]] = 2
        self.parameter["reference_answer"] = " ".join(map(str, labels))
    

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

            labels = processed_result
            if len(labels) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not all(label in (1, 2) for label in labels) :
                return self.rewards["invalid_solution"]
            
            degrees = [0] * self.parameter["N"]
            for u, v in self.parameter["edges"] :
                degrees[u] += (labels[u] == labels[v])
                degrees[v] += (labels[u] == labels[v])
            
            satisfied = sum(degree % 2 == 0 for degree in degrees)
            assert satisfied <= self.parameter["N"], "satisfied should be less than or equal to N"
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == self.parameter["N"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]