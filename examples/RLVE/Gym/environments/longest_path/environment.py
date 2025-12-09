import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class LongestPath_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a **directed graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following directed edges. Each edge is represented as a tuple `(s, t, w)`, meaning there is a directed edge **from vertex `s` to vertex `t` with weight `w`** :
{edges}

Your task is to find a path `p1, p2, ..., pk` such that:
- **No vertex appears more than once** in the path.
- Try your best to **maximize** the total weight of the path (i.e., the sum of all edge weights used).

**Output Format:** Your final answer should be a single line containing the path in order: `p1 p2 ... pk`, separated by **spaces**.
Example: `0 1 {N_minus_1}` (do **NOT** include the backticks or quotes); this means the path (k = 3) goes from `0` to `1` to `{N_minus_1}`."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the LongestPath_Environment instance.
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

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 < edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        edges = self.parameter["edges"] = random.sample([(s, t, random.randint(1, N)) for s in range(N) for t in range(N) if s != t], int(edge_density * N * (N - 1)))
        random.shuffle(edges)
        assert len(edges)

        assert len(edges) == len(set((s, t) for s, t, w in edges)), "edges should be unique"
        for s, t, w in edges :
            assert 0 <= s < N, "s should be in range"
            assert 0 <= t < N, "t should be in range"
            assert s != t, "s should not be equal to t"
        

        adjacent = [[] for s in range(N)]
        for s, t, w in edges :
            adjacent[s].append((t, w))

        self.parameter["gold_answer"] = 0
        dpF = dict()
        def dp(s : int, visited : int) -> int :
            if visited == (1 << N) - 1 :
                return 0
            if (s, visited) in dpF :
                return dpF[(s, visited)]
            ans = 0
            for t, w in adjacent[s] :
                if visited & (1 << t) == 0 :
                    ans = max(ans, dp(t, visited | (1 << t)) + w)
            dpF[(s, visited)] = ans
            return ans
        for s in range(N) :
            self.parameter["gold_answer"] = max(self.parameter["gold_answer"], dp(s, 1 << s))
        

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {}, {})".format(s, t, w) for s, t, w in self.parameter["edges"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                if not answer_array :
                    return None
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            path = processed_result
            if not all(0 <= vertex < self.parameter["N"] for vertex in path) :
                return self.rewards["invalid_solution"]
            if len(path) != len(set(path)) :
                return self.rewards["invalid_solution"]
            
            edge2weight = {(s, t) : w for s, t, w in self.parameter["edges"]}
            answer_weight = 0
            for s, t in zip(path, path[1 :]) :
                if (s, t) not in edge2weight :
                    return self.rewards["invalid_solution"]
                answer_weight += edge2weight[(s, t)]
            gold = self.parameter["gold_answer"]
            assert answer_weight <= gold and gold > 0, "answer_weight should be less than or equal to gold and gold should be greater than 0"

            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer_weight / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer_weight)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]