import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinimumWeightedSpanningTree_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning an undirected edge **connecting vertex `u` to vertex `v` with weight `w`**:
{edges}

Your task is to select a subset of edges `T = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)]` such that:
- `k = {N} - 1 = {N_minus_1}` (i.e., you select exactly {N_minus_1} edges).
- The selected edges form a **spanning tree** — that is, they connect all {N} vertices without forming any cycles.
- You choose one vertex as the **root**. Then, every non-root vertex has exactly one incoming edge in the tree.

The cost of your scheme (the edge subset and chosen root) is defined as follows:
- For each vertex `t ≠ root`, suppose `(s, t, w)` is the single incoming edge on the path from the root to `t`, and the number of edges from the root to `t` is `K`.
- The cost of this edge is `w × K`.
- The total cost is the sum of such edge costs for all `t ≠ root`.

Your goal is to **minimize the total cost** as defined above.

**Output Format:**
Output a single line containing the root and the endpoints of the selected edges in order: `root u_1 v_1 u_2 v_2 ... u_k v_k`, separated by **spaces** . Example: `0 0 1 1 2 1 3` (do **NOT** include the backticks or quotes); this means the root is `0`, and the selected edges are `(0, 1)`, `(1, 2)`, and `(1, 3)` (assuming 4 vertices in total)."""


    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinimumWeightedSpanningTree_Environment instance.
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
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        edges = self.parameter["edges"] = []

        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v, random.randint(0, N)))
        
        num_edges = int(edge_density * N * (N - 1) / 2)
        if len(edges) < num_edges :
            remaining_edges = list(set((u, v) for u in range(N) for v in range(u + 1, N)) - set((u, v) for u, v, w in edges))
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for u, v in remaining_edges :
                edges.append((u, v, random.randint(0, N)))
        random.shuffle(edges)

        for u, v, w in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, w in edges)), "edges should be unique"


        total_length = sum(w for u, v, w in edges)

        # A safe INF larger than any possible total cost
        INF = total_length * N + 1

        # Build adjacency matrix A
        A = [[INF] * N for _ in range(N)]
        for x, y, v in edges:
            if v < A[x][y]:
                A[x][y] = A[y][x] = v

        S = (1 << N) - 1

        # Precompute low‐bit index
        lg = [0] * (S + 1)
        for i in range(N):
            lg[1 << i] = i

        # f[i][j] = min cost to attach subset j (disjoint from i) to i by exactly |j| edges
        f = [dict() for _ in range(S + 1)]

        # *** FIX: make f[0][j] = 0 for all j, just like the C++ static init ***
        f[0] = {j: 0 for j in range(S + 1)}

        # Base case: attaching an empty set costs 0
        for i in range(1, S + 1):
            f[i][0] = 0

        ne = [0] * (S + 1)
        # Build f table
        for i in range(1, S + 1):
            s = S ^ i
            prev = 0
            j = s
            # build reverse linked list of submasks of s
            while j:
                ne[j] = prev
                prev = j
                j = (j - 1) & s

            # traverse that linked list
            j = prev
            while j:
                x = lg[j & -j]
                # find cheapest edge from x into i
                best = INF
                tmp = i
                while tmp:
                    yb = tmp & -tmp
                    y = lg[yb]
                    if A[x][y] < best:
                        best = A[x][y]
                    tmp ^= yb

                without_low = j ^ (j & -j)
                f[i][j] = f[i][without_low] + best
                j = ne[j]

        # g[l][i] = min cost to excavate exactly the set i using l roads
        g = [[INF] * (S + 1) for _ in range(N + 1)]
        # with 0 roads, only singletons are free
        for i in range(N):
            g[0][1 << i] = 0

        # build g
        for l in range(1, N + 1):
            for i in range(1, S + 1):
                j = i
                while j:
                    prev_set = i ^ j
                    cost = g[l - 1][prev_set] + f[prev_set][j] * l
                    if cost < g[l][i]:
                        g[l][i] = cost
                    j = (j - 1) & i

        # answer is min over all l
        ans = min(g[l][S] for l in range(N + 1))
        self.parameter["gold_answer"] = ans
    

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

            root = processed_result[0]
            if not (0 <= root < self.parameter["N"]) :
                return self.rewards["invalid_solution"]

            mst = processed_result[1 :]
            if len(mst) % 2 != 0 :
                return self.rewards["wrong_format"]
            mst = [(mst[i], mst[i + 1]) for i in range(0, len(mst), 2)]
            
            if len(mst) != self.parameter["N"] - 1 :
                return self.rewards["invalid_solution"]
            if not ((set(u for u, v in mst) | set(v for u, v in mst)) == set(range(self.parameter["N"]))) :
                return self.rewards["invalid_solution"]

            subgraph = networkx.Graph()
            edge2weight = {(u, v) : w for u, v, w in self.parameter["edges"]}
            for u, v in mst :
                u, v = min(u, v), max(u, v)
                if (u, v) not in edge2weight :
                    return self.rewards["invalid_solution"]
                subgraph.add_edge(u, v)
            if not networkx.is_connected(subgraph) :
                return self.rewards["invalid_solution"]
            assert networkx.is_tree(subgraph), "The answer should be a tree as it has N - 1 edges and is connected"
            
            answer_weight = 0
            adjacent_list = [[] for _ in range(self.parameter["N"])]
            for u, v in mst :
                adjacent_list[u].append(v)
                adjacent_list[v].append(u)
            def DFS(vertex : int, parent : int, depth : int) -> None :
                nonlocal answer_weight
                for neighbor in adjacent_list[vertex] :
                    if neighbor == parent :
                        continue
                    edge_weight = edge2weight[(min(vertex, neighbor), max(vertex, neighbor))]
                    answer_weight += edge_weight * (depth + 1)
                    DFS(neighbor, vertex, depth + 1)
            DFS(root, -1, 0)
            assert self.parameter["gold_answer"] <= answer_weight, "answer_weight should be greater than or equal to gold_answer"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if answer_weight == 0 :
                    assert self.parameter["gold_answer"] == 0, "If answer_weight is 0, gold_answer should also be 0"
                    return self.rewards["rewarding_weight"] * 1.0
                return self.rewards["rewarding_weight"] * ((self.parameter["gold_answer"] / answer_weight) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == answer_weight)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]