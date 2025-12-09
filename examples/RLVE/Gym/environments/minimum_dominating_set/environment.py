import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Minimum_DominatingSet_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices labeled from 0 to {N_minus_1}. The graph contains the following undirected edges:
{edges}

Each vertex has a cost, given as a list `C` of length {N}, where `C[i]` is the cost of vertex i:
{C}

Your task is to select a set of distinct vertices x_1, x_2, ..., x_k (you determine k), such that every vertex is either selected or has at least one selected neighbor.
Try your best to minimize the total cost: C[x_1] + C[x_2] + ... + C[x_k].

**Output Format:** Your final answer should be a single line containing the selected vertices in any order, separated by spaces.
Example: `0 1 {N_minus_1}` (do **NOT** include quotes or backticks)."""

    def __init__(self,
                 cost_range : int = 10,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Minimum_DominatingSet_Environment instance.
        """
        super().__init__(**kwargs)

        self.cost_range = cost_range
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
        assert N >= 2, "N should be greater than or equal to 1"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        edges = self.parameter["edges"] = random.sample([(u, v) for u in range(N) for v in range(u + 1, N)], int(edge_density * N * (N - 1) / 2))
        random.shuffle(edges)
        
        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"

        C = self.parameter["C"] = [random.randint(1, self.cost_range) for vertex in range(N)]


        covering = self.parameter["covering"] = [1 << u for u in range(N)]
        for u, v in edges :
            covering[u] |= 1 << v
            covering[v] |= 1 << u

        self.parameter["reference_answer"] = list(range(N))
        self.parameter["gold_answer"] = sum(C)

        selected = []
        def DFS(u : int, now_covering : int, sumC : int) -> None :
            if sumC >= self.parameter["gold_answer"] :
                return
            if u == N :
                if now_covering == (1 << N) - 1 :
                    assert sumC < self.parameter["gold_answer"]
                    self.parameter["reference_answer"], self.parameter["gold_answer"] = selected.copy(), sumC
                return
            DFS(u + 1, now_covering, sumC)
            if (now_covering | covering[u]) > now_covering :
                selected.append(u)
                DFS(u + 1, now_covering | covering[u], sumC + C[u])
                selected.pop()
        DFS(0, 0, 0)

        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["reference_answer"]))
        assert self.parameter["gold_answer"] > 0, "gold_answer must be greater than 0"

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
            C = "\n".join("C[{}]={}".format(i, Ci) for i, Ci in enumerate(self.parameter["C"])),
        )
    
    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            selected_vertices = processed_result

            if len(selected_vertices) != len(set(selected_vertices)) :
                return self.rewards["invalid_solution"]

            all_covering = 0
            for u in selected_vertices :
                if not (0 <= u < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                all_covering |= self.parameter["covering"][u]
            if all_covering != (1 << self.parameter["N"]) - 1 :
                return self.rewards["invalid_solution"]

            answer = sum(self.parameter["C"][u] for u in selected_vertices)
            gold = self.parameter["gold_answer"]
            assert gold <= answer, "gold should be less than or equal to answer"

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]