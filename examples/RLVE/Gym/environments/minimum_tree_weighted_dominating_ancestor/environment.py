import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinimumTreeWeightedDominatingAncestor_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3354
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} + 1 = {N_plus_1} vertices, labeled from `0` to `{N}`.

`0` is the root of the tree. Each non-root vertex has a parent, and the edge connecting it with its parent has a weight. The edges are given as follows:
{edges}

Each non-root vertex also has a cost, given as a list `C` of length {N}, where `C[i]` is the cost of vertex `i`:
{C}

The root (vertex `0`) is already selected. Your task is to select exactly {K} **additional** non-root vertices. The total cost of the selection is defined as follows:
- For every vertex `u`, let `D[u]` be the distance from `u` to its **nearest selected ancestor**, where a selected ancestor includes `0` or the vertex itself (if selected). The **distance** between two vertices is the sum of weights along the unique path between them.
- The cost contributed by vertex `u` is `C[u] × D[u]` for all non-root verticies `u`.
- Try your best to **minimize** the total cost.

**Output Format:** Output a single line containing {K} integers — the selected vertices (excluding 0), separated by spaces."""

    def __init__(self,
                 weight_range : int = 10,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinimumTreeWeightedDominatingAncestor_Environment instance.
        """
        super().__init__(**kwargs)

        self.weight_range = weight_range
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

        parents = self.parameter["parents"] = [None] * (N + 1)
        permutations = list(range(1, N + 1))
        random.shuffle(permutations)
        permutations = [0] + permutations
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            parent = random.choice(permutations[: index])
            parents[vertex] = (parent, random.randint(1, self.weight_range))
        
        C = self.parameter["C"] = [0] + [random.randint(0, self.weight_range) for vertex in range(1, N + 1)]

        K = self.parameter["K"] = random.randint(1, N - 1)


        graph = [[] for _ in range(N + 1)]
        # depth[i] = distance from root (Bytetown, node 0) to i
        depth = [0] * (N + 1)

        for i in range(1, N + 1):
            parent, dist = parents[i]
            graph[parent].append((i, dist))

        # f[p][j][l]: minimum cost in subtree of p, for wood from all nodes
        # that are descendants of p but no closer ancestor than j,
        # using l new sawmills in that subtree
        # g[p][j][l]: same but requiring that none of those l sawmills lies
        # on the path from p up to j (i.e., the first mill is strictly below p)
        f = [[[0] * (K + 1) for _ in range(N + 1)] for _ in range(N + 1)]
        g = [[[0] * (K + 1) for _ in range(N + 1)] for _ in range(N + 1)]

        # st is the stack of ancestors of the current node in DFS
        st = []

        def dfs(p):
            st.append(p)
            # Process children
            for to, dist in graph[p]:
                depth[to] = depth[p] + dist
                dfs(to)
                # Merge DP from child 'to' into p
                for j in st:
                    # We go from high to low l so we can use previous-state values safely
                    for l in range(K, -1, -1):
                        # First, take the case x = 0 (no new mills in 'to' subtree)
                        f[p][j][l] += f[to][j][0]
                        g[p][j][l] += f[to][p][0]
                        best_fpjl = f[p][j][l]
                        best_gpjl = g[p][j][l]
                        # Try allocating x new mills to subtree 'to'
                        for x in range(1, l + 1):
                            # put x mills in 'to' subtree for wood going up to j
                            cost_f = f[p][j][l - x] + f[to][j][x]
                            if cost_f < best_fpjl:
                                best_fpjl = cost_f
                            # put x mills in 'to' subtree for wood going up to p
                            cost_g = g[p][j][l - x] + f[to][p][x]
                            if cost_g < best_gpjl:
                                best_gpjl = cost_g
                        f[p][j][l] = best_fpjl
                        g[p][j][l] = best_gpjl

            # After merging all children, account for p's own wood
            for j in st:
                dist_up = depth[p] - depth[j]
                # if no new mills in subtree of p, we pay full transport cost
                f[p][j][0] += C[p] * dist_up
                # if we have at least 1 mill, we can choose either to treat at p
                # or let it be handled by one of the mills below p
                for l in range(1, K + 1):
                    f[p][j][l] = min(
                        f[p][j][l] + C[p] * dist_up,
                        g[p][j][l - 1]
                    )

            st.pop()

        # Run DFS from root (Bytetown = 0)
        dfs(0)
        # Answer: minimum cost when we place exactly K new mills in the whole tree
        self.parameter["gold_answer"] = f[0][0][K]
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_plus_1 = N + 1,
            edges = "\n".join("`{}`'s parent is `{}` with weight `{}`".format(i + 1, parent, weight) for i, (parent, weight) in enumerate(self.parameter["parents"][1 :])),
            C = " ".join("C[{}]={}".format(i, Ci) for i, Ci in enumerate(self.parameter["C"]) if i > 0),
            K = self.parameter["K"],
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

            if len(selected_vertices) != self.parameter["K"] :
                return self.rewards["invalid_solution"]
            
            selected = [True] + [False] * self.parameter["N"]
            for vertex in selected_vertices :
                if not (1 <= vertex <= self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                selected[vertex] = True
            
            graph = [[] for _ in range(self.parameter["N"] + 1)]
            for i in range(1, self.parameter["N"] + 1):
                parent, dist = self.parameter["parents"][i]
                graph[parent].append((i, dist))
            
            answer = 0
            def DFS(vertex, dist) :
                nonlocal answer
                if selected[vertex] :
                    dist = 0
                answer += self.parameter["C"][vertex] * dist
                for neighbor, weight in graph[vertex] :
                    DFS(neighbor, dist + weight)
            DFS(0, 0)
            gold = self.parameter["gold_answer"]
            assert gold <= answer, "gold should be less than or equal to answer, but got gold={} and answer={}".format(gold, answer)

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if answer == 0 :
                    assert gold == 0, "If answer is 0, gold should also be 0"
                    return self.rewards["rewarding_weight"] * 1.0
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]