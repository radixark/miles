import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MaximumAchromaticNumber_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.
The graph contains the following undirected edges:
{edges}

Your task is to assign a **non-negative integer color** to each vertex, represented as `c[0], c[1], ..., c[{N_minus_1}]`, such that:
- For every edge `(u, v)` in the graph, `c[u] ≠ c[v]` — adjacent vertices must have different colors.
- For every pair of two distinct used colors `x` and `y`, there exists **at least one edge** `(u, v)` such that `c[u] = x` and `c[v] = y`, i.e., this is a *complete coloring*.
- The total number of **distinct colors used** (i.e., the number of unique values among `c[0]` to `c[{N_minus_1}]`) is **maximized** - try your best to find a valid coloring using as many colors as possible.

**Output Format:**
Your final answer should be a single line containing the color of each vertex in order: `c[0], c[1], ..., c[{N_minus_1}]`, separated by **spaces**."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MaximumAchromaticNumber_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
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


        self.parameter["reference_answer"] = None
        self.parameter["gold_answer"] = 0

        adjacent = [0] * N
        smaller_adjacents = [[] for u in range(N)]
        for u, v in edges :
            adjacent[u] |= 1 << v
            adjacent[v] |= 1 << u
            smaller_adjacents[max(u, v)].append(min(u, v))
        
        colors, color2set = [None] * N, [0] * N
        def DFS(u : int, max_color : int) -> int :
            if (max_color + 1) + (N - u) <= self.parameter["gold_answer"] :
                return
            if u == N :
                color_adjacent = [[False] * (max_color + 1) for _ in range(max_color + 1)]
                satisfied_color_pair_num = 0
                for u, v in edges :
                    color_u, color_v = min(colors[u], colors[v]), max(colors[u], colors[v])
                    assert color_u != color_v, "Adjacent vertices should have different colors"
                    if not color_adjacent[color_u][color_v] :
                        color_adjacent[color_u][color_v] = True
                        satisfied_color_pair_num += 1
                assert satisfied_color_pair_num <= (max_color + 1) * max_color // 2, "The number of satisfied color pairs should not exceed the maximum possible pairs"
                if satisfied_color_pair_num == (max_color + 1) * max_color // 2 :
                    self.parameter["reference_answer"], self.parameter["gold_answer"] = colors.copy(), max_color + 1
                return
            for color in range((max_color + 1) + 1) :
                if (color2set[color] & adjacent[u]) == 0 :
                    colors[u] = color
                    color2set[color] += 1 << u
                    DFS(u + 1, max(max_color, color))
                    color2set[color] -= 1 << u
        DFS(0, -1)

        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["reference_answer"]))
    

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
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            colors = processed_result
            if len(colors) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            adjacent_color_pairs = set()
            for u, v in self.parameter["edges"] :
                if colors[u] == colors[v] :
                    return self.rewards["invalid_solution"]
                adjacent_color_pairs.add((min(colors[u], colors[v]), max(colors[u], colors[v])))
            
            assert len(adjacent_color_pairs) <= len(set(colors)) * (len(set(colors)) - 1) // 2, "The number of adjacent color pairs should not exceed the maximum possible pairs"
            if len(adjacent_color_pairs) < len(set(colors)) * (len(set(colors)) - 1) // 2 :
                return self.rewards["invalid_solution"]
            
            gold, answer = self.parameter["gold_answer"], len(set(colors))
            assert answer <= gold, "The number of distinct colors used should not exceed the gold answer"
            
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]