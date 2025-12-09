import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinimumHarmoniousChromaticNumber_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.
The graph contains the following undirected edges:
{edges}

Your task is to assign a **non-negative integer color** to each vertex, represented as `c[0], c[1], ..., c[{N_minus_1}]`, such that:
- For every edge `(u, v)` in the graph, `c[u] ≠ c[v]` — adjacent vertices must have different colors.
- For every pair of two distinct used colors `x` and `y`, there exists **at most one edge** `(u, v)` such that `c[u] = x` and `c[v] = y`, i.e., this is a *harmonious coloring*.
- The total number of **distinct colors used** (i.e., the number of unique values among `c[0]` to `c[{N_minus_1}]`) is **minimized** - try your best to find a valid coloring using as few colors as possible.

**Output Format:**
Your final answer should be a single line containing the color of each vertex in order: `c[0], c[1], ..., c[{N_minus_1}]`, separated by **spaces**."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinimumHarmoniousChromaticNumber_Environment instance.
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


        self.parameter["reference_answer"] = list(range(N))
        self.parameter["gold_answer"] = N

        adjacent = [0] * N
        smaller_adjacents = [[] for u in range(N)]
        for u, v in edges :
            adjacent[u] |= 1 << v
            adjacent[v] |= 1 << u
            smaller_adjacents[max(u, v)].append(min(u, v))
        
        colors, color2set = [None] * N, [0] * N
        color_adjacent = [[False] * N for _ in range(N)]
        def DFS(u : int, max_color : int) -> int :
            nonlocal colors, color2set, color_adjacent
            if max_color + 1 >= self.parameter["gold_answer"] :
                return
            if u == N :
                self.parameter["reference_answer"], self.parameter["gold_answer"] = colors.copy(), max_color + 1
                return
            for color in range((max_color + 1) + 1) :
                if (color2set[color] & adjacent[u]) == 0 :
                    colors[u] = color

                    new_color_adjacent = [row.copy() for row in color_adjacent]

                    invalid = False
                    for v in smaller_adjacents[u] :
                        color_u, color_v = min(colors[u], colors[v]), max(colors[u], colors[v])
                        assert color_u != color_v, "Adjacent vertices should have different colors"
                        if new_color_adjacent[color_u][color_v] :
                            invalid = True
                            break
                        new_color_adjacent[color_u][color_v] = True
                    
                    if not invalid :
                        color2set[color] += 1 << u
                        old_color_adjacent = [row.copy() for row in color_adjacent]
                        color_adjacent = new_color_adjacent
                        DFS(u + 1, max(max_color, color))
                        color_adjacent = old_color_adjacent
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
                color_u, color_v = min(colors[u], colors[v]), max(colors[u], colors[v])
                if (color_u, color_v) in adjacent_color_pairs :
                    return self.rewards["invalid_solution"]
                adjacent_color_pairs.add((color_u, color_v))
            
            gold, answer = self.parameter["gold_answer"], len(set(colors))
            assert gold <= answer, "gold should be less than or equal to answer"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]