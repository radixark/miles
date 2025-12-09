import random
from collections import deque
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class CirculatingGrid_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3965
    prompt_template = \
r"""Consider a {R} × {C} grid, where each cell has coordinates (i, j) (0 ≤ i < {R}, 0 ≤ j < {C}). Each cell contains one of the characters `L`, `R`, `U`, or `D`, meaning:
- `L`: moves to (i, (j - 1) MOD {C})
- `R`: moves to (i, (j + 1) MOD {C})
- `U`: moves to ((i - 1) MOD {R}, j)
- `D`: moves to ((i + 1) MOD {R}, j)
Here, (-1 MOD N) = N - 1.

You are given such a grid:
{grid}

Modify any number of cells so that the resulting grid satisfies the following condition: Starting from any cell, it must be possible to eventually return to the same cell (simply standing there at the beginning does not count). Can you use as small the number of changes (i.e., number of cells modified) as possible? Output the modified grid in the same format — exactly {R} lines, each containing {C} characters (`L`, `R`, `U`, or `D`) with **no separators**."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the CirculatingGrid_Environment instance.
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
        assert "MAX_R_C" in self.parameter, "MAX_R_C is required in parameter"
        MAX_R_C = self.parameter["MAX_R_C"]
        assert MAX_R_C >= 3, "MAX_R_C must be at least 3"

        R, C = self.parameter["R"], self.parameter["C"] = random.randint(2, MAX_R_C), random.randint(2, MAX_R_C)
        
        LRUD_distribution = [random.randint(1, R * C) for _ in range(4)]
        grid = self.parameter["grid"] = [[random.choices(['L', 'R', 'U', 'D'], weights = LRUD_distribution)[0] for _ in range(C)] for _ in range(R)]


        # Directions: L, R, U, D
        DX = [0, 0, -1, 1]   # row delta
        DY = [-1, 1, 0, 0]   # col delta
        DIR_ID = {'L': 0, 'R': 1, 'U': 2, 'D': 3}

        class Edge:
            __slots__ = ('to', 'rev', 'cap', 'cost')
            def __init__(self, to, rev, cap, cost):
                self.to = to
                self.rev = rev
                self.cap = cap
                self.cost = cost

        def add_edge(graph, u, v, cap, cost):
            graph[u].append(Edge(v, len(graph[v]), cap, cost))
            graph[v].append(Edge(u, len(graph[u]) - 1, 0, -cost))

        def min_cost_max_flow(graph, N, s, t, INF):
            flow = 0
            cost = 0
            dist = [0] * N
            inq = [False] * N
            prev_node = [-1] * N
            prev_edge = [-1] * N

            while True:
                # SPFA to find shortest augmenting path by cost
                for i in range(N):
                    dist[i] = INF
                    inq[i] = False
                    prev_node[i] = -1
                    prev_edge[i] = -1
                dist[s] = 0
                q = deque([s])
                inq[s] = True

                while q:
                    u = q.popleft()
                    inq[u] = False
                    for ei, e in enumerate(graph[u]):
                        if e.cap > 0:
                            v = e.to
                            nd = dist[u] + e.cost
                            if nd < dist[v]:
                                dist[v] = nd
                                prev_node[v] = u
                                prev_edge[v] = ei
                                if not inq[v]:
                                    inq[v] = True
                                    q.append(v)

                if prev_node[t] == -1:
                    break  # no more augmenting paths

                # Find bottleneck
                addf = INF
                v = t
                while v != s:
                    u = prev_node[v]
                    ei = prev_edge[v]
                    e = graph[u][ei]
                    if e.cap < addf:
                        addf = e.cap
                    v = u

                # Augment
                v = t
                while v != s:
                    u = prev_node[v]
                    ei = prev_edge[v]
                    e = graph[u][ei]
                    e.cap -= addf
                    graph[v][e.rev].cap += addf
                    cost += addf * e.cost
                    v = u

                flow += addf

            return flow, cost

        def compute():
            # MP holds the direction id (0..3) for each cell
            MP = [[0] * C for _ in range(R)]
            for i in range(R):
                for j in range(C):
                    MP[i][j] = DIR_ID[grid[i][j]]

            n_left = R * C
            offset = n_left
            s = 2 * n_left
            t = s + 1
            N = t + 1

            # INF derived from input size; safely larger than any possible path cost
            INF = R * C * 4 + 5

            graph = [[] for _ in range(N)]

            # Build edges from each cell (left partition) to its 4 neighbors (right partition)
            for i in range(R):
                for j in range(C):
                    u = i * C + j
                    for k in range(4):
                        ni = (i + DX[k]) % R
                        nj = (j + DY[k]) % C
                        v = offset + (ni * C + nj)
                        cost = 0 if k == MP[i][j] else 1
                        add_edge(graph, u, v, 1, cost)

            # Source to all left nodes; all right nodes to sink
            for u in range(n_left):
                add_edge(graph, s, u, 1, 0)
            for v in range(offset, offset + n_left):
                add_edge(graph, v, t, 1, 0)

            _, total_cost = min_cost_max_flow(graph, N, s, t, INF)
            return total_cost
        
        self.parameter["gold_answer"] = compute()
        assert self.parameter["gold_answer"] >= 0, "Gold answer must be non-negative"
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            R = self.parameter["R"],
            C = self.parameter["C"],
            grid = "\n".join("".join(row) for row in self.parameter["grid"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[List[str]] :
        if answer is not None :
            answer = answer.strip()
            grid = []
            for line in answer.splitlines() :
                line = line.strip()
                if line :
                    grid.append(line)
            return grid
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            grid = processed_result

            if len(grid) != self.parameter["R"] :
                return self.rewards["wrong_format"]
            if not all(len(row) == self.parameter["C"] for row in grid) :
                return self.rewards["wrong_format"]
            if not all(all(c in "LRUD" for c in row) for row in grid) :
                return self.rewards["wrong_format"]

            in_degree = [[0] * self.parameter["C"] for _ in range(self.parameter["R"])]
            for i in range(self.parameter["R"]) :
                for j in range(self.parameter["C"]) :
                    if grid[i][j] == "L" :
                        in_degree[i][(j - 1 + self.parameter["C"]) % self.parameter["C"]] += 1
                    elif grid[i][j] == "R" :
                        in_degree[i][(j + 1) % self.parameter["C"]] += 1
                    elif grid[i][j] == "U" :
                        in_degree[(i - 1 + self.parameter["R"]) % self.parameter["R"]][j] += 1
                    elif grid[i][j] == "D" :
                        in_degree[(i + 1) % self.parameter["R"]][j] += 1
                    else :
                        assert False, "Invalid character in grid"
            if not all(in_degree[i][j] == 1 for i in range(self.parameter["R"]) for j in range(self.parameter["C"])) :
                return self.rewards["invalid_solution"]

            answer, gold = sum(int(grid[i][j] != self.parameter["grid"][i][j]) for i in range(self.parameter["R"]) for j in range(self.parameter["C"])), self.parameter["gold_answer"]
            assert gold <= answer, "Gold answer is greater than the computed answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if answer == 0 :
                    assert gold == 0, "Gold answer is non-zero but computed answer is zero"
                    return self.rewards["rewarding_weight"] * 1.0
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]