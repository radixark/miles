import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinimumRatioPath_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2502
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning there is an undirected edge connecting vertex `u` and vertex `v` with weight `w`:
{edges}

Your task is to find a path `p1, p2, ..., pk` such that:
- `p1 = 0` (the path starts at vertex `0`)
- `pk = {N_minus_1}` (the path ends at vertex `{N_minus_1}`)
- Try your best to **minimize** the ratio of the maximum edge weight to the minimum edge weight along the path (i.e., minimize `max(w) / min(w)`, where `w` are the edge weights on the path).

**Output Format:** Your final answer should be a single line containing the path in order: `p1 p2 ... pk`, separated by spaces. Example: `0 1 {N_minus_1}` (do NOT include backticks or quotes)."""

    def __init__(self,
                 weight_range_multiple : int = 5,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinimumRatioPath_Environment instance.
        """
        super().__init__(**kwargs)

        self.weight_range_multiple = weight_range_multiple

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

        assert "edge_ratio" in self.parameter, "edge_ratio is required in parameter"
        edge_ratio = self.parameter["edge_ratio"]

        edges = self.parameter["edges"] = []
        
        constructed_path = list(range(1, (N - 2) + 1))
        random.shuffle(constructed_path)
        constructed_path = [0] + constructed_path + [N - 1]
        assert set(constructed_path) == set(range(N)), "constructed_path should contain all vertices from 0 to N-1"
        for u, v in zip(constructed_path, constructed_path[1 :]) :
            w = random.randint(1, max(1, int(N * edge_ratio) * self.weight_range_multiple))
            edges.append((min(u, v), max(u, v), w))
        
        num_edges = int(N * edge_ratio)
        if len(edges) < num_edges :
            remaining_edges = list(set((u, v) for u in range(N) for v in range(u + 1, N) if (u, v) != (0, N - 1)) - set((u, v) for u, v, w in edges))
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for u, v in remaining_edges :
                edges.append((u, v, random.randint(1, max(1, int(N * edge_ratio) * self.weight_range_multiple))))
        random.shuffle(edges)

        for u, v, w in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, w in edges)), "edges should be unique"


        edges = sorted([(w, u, v) for u, v, w in edges], key = lambda x : x[0])
        M = len(edges)
        S, T = 0, N - 1  # Start and end vertices

        ans_num = 0   # numerator = max speed on the chosen path
        ans_den = 1   # denominator = min speed on the chosen path
        found_any = False

        def find(parent, x):
            if parent[x] != x:
                parent[x] = find(parent, parent[x])
            return parent[x]

        # Try every possible minimum-speed edge as the start of the path
        for i in range(M):
            parent = list(range(N))
            # Add edges in non-decreasing order of speed, starting from i,
            # until s and t become connected
            for j in range(i, M):
                wj, uj, vj = edges[j]
                fu = find(parent, uj)
                fv = find(parent, vj)
                if fu != fv:
                    parent[fu] = fv
                if find(parent, S) == find(parent, T):
                    break

            # If even after adding all edges from i onward s and t aren't connected:
            if find(parent, S) != find(parent, T):
                if i == 0:
                    assert False
                break

            wi = edges[i][0]  # the minimum speed on this trial
            # Update the best ratio if it's the first valid path, or if the new ratio is smaller:
            #   compare ans_num/ans_den  >=  wj/wi  〈⇒〉  ans_num * wi >= ans_den * wj
            if not found_any or ans_num * wi >= ans_den * wj:
                ans_num = wj
                ans_den = wi
                found_any = True

        self.parameter["gold_answer"] = (ans_num, ans_den)
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {}, {})".format(u, v, w) for u, v, w in self.parameter["edges"]),
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
            
            edge2weight = {(u, v) : w for u, v, w in self.parameter["edges"]}
            answer_num, answer_den = min(edge2weight.values()), max(edge2weight.values())
            for s, t in zip(path, path[1 :]) :
                u, v = min(s, t), max(s, t)
                if (u, v) not in edge2weight :
                    return self.rewards["invalid_solution"]
                w = edge2weight[(u, v)]
                answer_num, answer_den = max(answer_num, w), min(answer_den, w)
            gold_num, gold_den = self.parameter["gold_answer"]
            # gold_num / gold_den <= answer_num / answer_den <=> gold_num * answer_den <= answer_num * gold_den
            assert gold_num * answer_den <= answer_num * gold_den, "The answer should be better than the gold answer"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                # (gold_num / gold_den) / (answer_num / answer_den) = (gold_num * answer_den) / (answer_num * gold_den)
                return self.rewards["rewarding_weight"] * (((gold_num * answer_den) / (answer_num * gold_den)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * ((gold_num * answer_den) == (answer_num * gold_den))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]