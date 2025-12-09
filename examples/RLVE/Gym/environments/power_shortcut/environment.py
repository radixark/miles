import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class PowerShortcut_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a **directed graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following directed edges. Each edge is represented as a tuple `(s, t)`, meaning there is a directed edge **from vertex `s` to vertex `t`**:
{edges}

Your task is to find a sequence of vertices `p[1], p[2], ..., p[m]` such that:
- `p[1] = 0` (the sequence starts at vertex 0) and `p[m] = {N_minus_1}` (the sequence ends at vertex `{N_minus_1}`)
- For each consecutive pair `(p[i], p[i + 1])`, there exists a **path** from `p[i]` to `p[i + 1]` whose length (number of edges) is exactly 2^k for some integer k where 0 ≤ k ≤ {K}.

Your goal is to **minimize** the length `m` of the sequence — that is, the number of steps in the sequence.

**Output Format:**
Your final answer should be a single line containing the sequence: `p[1] p[2] ... p[m]`, separated by **spaces**.
Example: `0 1 {N_minus_1}` (do **NOT** include the backticks or quotes); this means m = 3, p[1] = 0, p[2] = 1, and p[3] = {N_minus_1}."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the PowerShortcut_Environment instance.
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

        assert "K" in self.parameter, "K is required in parameter"
        K = self.parameter["K"]
        assert K >= 0, "K should be greater than or equal to 0"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        constructed_path = list(range(1, N - 1))
        random.shuffle(constructed_path)
        constructed_path = [0] + constructed_path + [N - 1]

        edges = self.parameter["edges"] = []
        for s, t in zip(constructed_path, constructed_path[1 :]) :
            edges.append((s, t))
        
        num_edges = int(edge_density * N * (N - 1))
        if len(edges) < num_edges :
            remaining_edges = list(set((s, t) for s in range(N) for t in range(N) if s != t) - set(edges))
            edges += random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
        random.shuffle(edges)

        assert len(edges) == len(set(edges)), "Edges should be unique"
        for s, t in edges :
            assert 0 <= s < N, "s should be in range"
            assert 0 <= t < N, "t should be in range"
            assert s != t, "s should not be equal to t"


        achievable = [[[False] * N for s in range(N)] for k in range(K + 1)]
        path = [[None] * N for s in range(N)]
        for s in range(N) :
            path[s][s] = []
        for s, t in edges :
            achievable[0][s][t] = True
            path[s][t] = []
        for k in range(1, K + 1) :
            for s in range(N) :
                for t in range(N) :
                    for m in range(N) :
                        achievable[k][s][t] |= (achievable[k - 1][s][m] and achievable[k - 1][m][t])
                    if achievable[k][s][t] :
                        path[s][t] = []
        self.parameter["achievable"] = [[any(achievable[k][s][t] for k in range(K + 1)) for t in range(N)] for s in range(N)]

        for m in range(N) :
            for s in range(N) :
                for t in range(N) :
                    if path[s][m] is not None and path[m][t] is not None :
                        if path[s][t] is None or (len(path[s][t]) > len(path[s][m]) + 1 + len(path[m][t])) :
                            path[s][t] = path[s][m] + [m] + path[m][t]
        self.parameter["reference_answer"] = " ".join(map(str, [0] + path[0][N - 1] + [N - 1]))
        self.parameter["gold_answer"] = 1 + len(path[0][N - 1]) + 1
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            K = self.parameter["K"],
            edges = "\n".join("({}, {})".format(s, t) for s, t in self.parameter["edges"]),
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
            for vertex in path :
                if not (0 <= vertex < self.parameter["N"]) : # check if vertex is in range
                    return self.rewards["invalid_solution"]
            if not (path[0] == 0 and path[-1] == self.parameter["N"] - 1) : # check if start and end vertices are correct
                return self.rewards["invalid_solution"]
            for s, t in zip(path, path[1 :]) :
                if not self.parameter["achievable"][s][t] :
                    return self.rewards["invalid_solution"]
            
            gold, answer = self.parameter["gold_answer"], len(path)
            assert gold <= answer, "gold_answer should be less than or equal to answer length"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]