import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class ShortestPath_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a **directed graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following directed edges. Each edge is represented as a tuple `(s, t, w)`, meaning there is a directed edge **from vertex `s` to vertex `t` with weight `w`** :
{edges}

Your task is to find a path `p1, p2, ..., pk` such that:
- `p1 = 0` (the path starts at vertex 0) and `pk = {N_minus_1}` (the path ends at vertex `{N_minus_1}`)
- Try your best to **minimize** the total weight of the path (i.e., the sum of all edge weights used).

**Output Format:**
Your final answer should be a single line containing the path in order: `p1 p2 ... pk`, separated by **spaces**.
Example: `0 1 {N_minus_1}` (do **NOT** include the backticks or quotes); this means the path (k = 3) goes from `0` to `1` to `{N_minus_1}`.
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the ShortestPath_Environment instance.
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
        assert N >= 4, "N should be greater than or equal to 4"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        edges = self.parameter["edges"] = []

        constructed_path = list(range(1, N - 1))
        random.shuffle(constructed_path)
        constructed_path = [0] + constructed_path + [N - 1]
        for s, t in zip(constructed_path, constructed_path[1 :]) :
            w = random.randint(1, max(1, N // 3))
            edges.append((s, t, w))
        
        num_edges = int(edge_density * N * (N - 1))
        if len(edges) < num_edges :
            remaining_edges = list(set((s, t) for s in range(N) for t in range(N) if s != t) - set((s, t) for s, t, w in edges) - {(0, N - 1)})
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for s, t in remaining_edges :
                edges.append((s, t, random.randint(max(1, N // 2), N)))
        random.shuffle(edges)

        starting = {t : (s, t, w) for s, t, w in edges if s == 0}
        ending = {s : (s, t, w) for s, t, w in edges if t == N - 1}
        for s, t, w in starting.values() :
            if t in ending :
                if t == constructed_path[-2] :
                    assert t != constructed_path[1]
                    edges.remove(starting[t])
                else :
                    edges.remove(ending[t])
            

        assert len(edges) == len(set((s, t) for s, t, w in edges)), "edges should be unique"
        for s, t, w in edges :
            assert 0 <= s < N, "s should be in range"
            assert 0 <= t < N, "t should be in range"
            assert s != t, "s should not be equal to t"
        

        G = networkx.DiGraph()
        G.add_weighted_edges_from(edges)
        shortest_path_length, shortest_path = networkx.single_source_dijkstra(G, 0, N - 1)
        self.parameter["reference_answer_weight"] = shortest_path_length
        self.parameter["reference_answer"] = " ".join(map(str, shortest_path))
    
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
            for vertex in path :
                if not (0 <= vertex < self.parameter["N"]) : # check if vertex is in range
                    return self.rewards["invalid_solution"]
            if not (path[0] == 0 and path[-1] == self.parameter["N"] - 1) : # check if start and end vertices are correct
                return self.rewards["invalid_solution"]
            
            edge2weight = {(s, t) : w for s, t, w in self.parameter["edges"]}
            answer_weight = 0
            for s, t in zip(path, path[1 :]) :
                if (s, t) not in edge2weight :
                    return self.rewards["invalid_solution"]
                answer_weight += edge2weight[(s, t)]
            gold = self.parameter["reference_answer_weight"]
            assert 0 < gold <= answer_weight, "answer weight should be greater than or equal to reference"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer_weight) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer_weight)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]