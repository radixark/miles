import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class ColoringCounting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The graph contains the following undirected edges:
{edges}

You are also given an array `R` of length {N}, where `R[u]` denotes the **maximum allowed color** for vertex `u`:
{R}

A coloring assigns an integer `C[u]` to each vertex `u`, satisfying the following conditions:
- `0 <= C[u] <= R[u]` for all vertices `u`
- For every edge `(u, v)`, `C[u] ≠ C[v]` (i.e., adjacent vertices must have different colors)

The **value** of a valid coloring is the number of **distinct colors used** (i.e., the count of unique values among `C[0], C[1], ..., C[{N_minus_1}]`). Please compute the **total value of all valid colorings**.

**Output Format:** Your final answer should be a single integer — the **sum of values** over all valid colorings of the graph."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the ColoringCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
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

        Deg = [0] * N
        
        for u, v in edges :
            assert 0 <= u < v < N
            Deg[u] += 1
            Deg[v] += 1
        assert len(edges) == len(set(edges)), "edges should be unique"

        R = self.parameter["R"] = tuple(random.randint(Deg[u], 2 * Deg[u]) for u in range(N))


        nodes = list(enumerate(R))
        nodes.sort(key = lambda x : x[1])
        sorted_R = [r for _, r in nodes]
        orig_to_sorted = [0] * N
        for new_idx, (orig_idx, _) in enumerate(nodes) :
            orig_to_sorted[orig_idx] = new_idx

        G = [[False] * N for _ in range(N)]
        for u, v in edges :
            u = orig_to_sorted[u]
            v = orig_to_sorted[v]
            G[u][v] = G[v][u] = True

        total_S = 1 << N
        Can = [True] * total_S
        for S in range(total_S) :
            for u in range(N) :
                if not (S >> u) & 1 :
                    continue
                for v in range(u + 1, N) :
                    if (S >> v) & 1 and G[u][v]:
                        Can[S] = False
                        break
                if not Can[S] :
                    break

        F = [[0] * (N + 1) for _ in range(total_S)]
        F[total_S - 1][0] = 1

        for S in range(total_S - 1, 0, -1) :
            for i in range(N) :
                if (S >> i) & 1 :
                    Min = i
                    break
            max_k = min(sorted_R[Min], N - 1)
            for k in range(max_k + 1) :
                ways = F[S][k]
                if ways == 0 :
                    continue
                W = S & ~(1 << Min)
                T = W
                while True :
                    if Can[T | (1 << Min)] :
                        new_S = W & ~T
                        F[new_S][k + 1] += ways * (sorted_R[Min] + 1 - k)
                    if T == 0 :
                        break
                    T = (T - 1) & W

        self.parameter["reference_answer"] = sum(F[0][k] * k for k in range(1, N + 1))
        assert self.parameter["reference_answer"] > 0
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
            R = "\n".join("R[{}]={}".format(u, Ru) for u, Ru in enumerate(self.parameter["R"])),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[int] :
        if answer is not None :
            answer = answer.strip()
            try :
                int_answer = int(answer)
                return int_answer
            except ValueError :
                return None
        else :
            return None
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result <= 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]