import heapq
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class HamiltonianPath_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a **directed graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following directed edges. Each edge is represented as a tuple `(s, t, w)`, meaning there is a directed edge **from vertex `s` to vertex `t` with weight `w`**:
{edges}

Your task is to find a path `p1, p2, ..., pk` such that:
- The path **visits every vertex at least once** (revisiting vertices is allowed).
- Your goal is to **minimize the total weight** of the path. The total weight is the sum of the weights of all edges used in the path.

Output Format:
Your final answer should be a single line containing the path in order: `p1, p2, ..., pk`, separated by **spaces**.
Example: `0 1 0 2` (do **NOT** include the backticks or quotes); this means the path starts at vertex 0, goes to 1, returns to 0, and then to 2 — thus visiting all three vertices at least once (assuming 3 vertices in total).
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the HamiltonianPath_Environment instance.
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
        assert N >= 2, "N should be greater than or equal to 2"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        edges = self.parameter["edges"] = []

        constructed_path = list(range(N))
        random.shuffle(constructed_path)
        self.parameter["reference_answer"] = " ".join(map(str, constructed_path))
        self.parameter["reference_answer_weight"] = 0
        for s, t in zip(constructed_path, constructed_path[1 :]) :
            w = random.randint(1, N)
            edges.append((s, t, w))
            self.parameter["reference_answer_weight"] += w
        
        num_edges = int(edge_density * N * (N - 1))
        if len(edges) < num_edges :
            remaining_edges = list(set((s, t) for s in range(N) for t in range(N) if s != t) - set((s, t) for s, t, w in edges))
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for s, t in remaining_edges :
                edges.append((s, t, random.randint(1, max(1, N // 2))))
        random.shuffle(edges)
        
        assert len(edges) == len(set((s, t) for s, t, w in edges)), "edges should be unique"
        for s, t, w in edges :
            assert 0 <= s < N, "s should be in range"
            assert 0 <= t < N, "t should be in range"
            assert s != t, "s should not be equal to t"
        

        adjacent = [[] for s in range(N)]
        for s, t, w in edges :
            adjacent[s].append((t, w))
        priority_queue = [(0, (1 << start, start)) for start in range(N)]
        visited_states, dist, prev = set(), {(1 << start, start) : 0 for start in range(N)}, {(1 << start, start) : (0, -1) for start in range(N)}

        while priority_queue :
            current_dist, (visited, s) = heapq.heappop(priority_queue)

            if visited == (1 << N) - 1 :
                assert current_dist < self.parameter["reference_answer_weight"], "current_dist should be less than or equal to reference_answer_weight"
                self.parameter["reference_answer_weight"] = current_dist

                self.parameter["reference_answer"] = []
                while True :
                    assert (visited == 0) == (s == -1), "visited should be 0 if and only if s is -1"
                    if visited == 0 :
                        break
                    self.parameter["reference_answer"].append(s)
                    visited, s = prev[(visited, s)]
                self.parameter["reference_answer"].reverse()
                self.parameter["reference_answer"] = " ".join(map(str, self.parameter["reference_answer"]))
                
                break

            if (visited, s) in visited_states :
                continue
            visited_states.add((visited, s))

            for t, w in adjacent[s] :
                new_visited = visited | (1 << t)
                new_dist = current_dist + w
                if dist.get((new_visited, t), self.parameter["reference_answer_weight"]) > new_dist :
                    dist[(new_visited, t)] = new_dist
                    prev[(new_visited, t)] = (visited, s)
                    heapq.heappush(priority_queue, (new_dist, (new_visited, t)))

    

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
            if len(set(path)) != self.parameter["N"] : # check if all vertices are visited
                return self.rewards["invalid_solution"]
            
            edge2weight = {(s, t) : w for s, t, w in self.parameter["edges"]}
            answer_weight = 0
            for s, t in zip(path, path[1 :]) :
                if (s, t) not in edge2weight :
                    return self.rewards["invalid_solution"]
                answer_weight += edge2weight[(s, t)]
            assert self.parameter["reference_answer_weight"] <= answer_weight, "answer weight should be greater than or equal to reference_answer_weight"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((self.parameter["reference_answer_weight"] / answer_weight) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer_weight == self.parameter["reference_answer_weight"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]