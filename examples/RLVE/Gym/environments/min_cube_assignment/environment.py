import random
from collections import deque
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinCubeAssignment_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3227
    prompt_template = \
r"""You are given a {P} × {Q} grid. You need to assign each cell (i, j) an integer value f(i, j) in the range [0, {R}). Each cell (i, j) contributes a cost of c(i, j, f(i, j)) to the total cost, where the cost function c is defined as:
{costs}

In addition, for every pair of **adjacent** cells (i, j) and (i', j') (i.e., cells such that |i - i'| + |j - j'| = 1), the assigned values must satisfy |f(i, j) - f(i', j')| ≤ {D}. Please find an assignment of values to the grid that minimizes the total cost.

**Output Format:** Output {P} lines, each with {Q} integers (space-separated), representing the values assigned to the grid in row-major order."""


    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinCubeAssignment_Environment instance.
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
        assert "MAX_P_Q_R" in self.parameter, "MAX_P_Q_R is required in parameter"
        MAX_P_Q_R = self.parameter["MAX_P_Q_R"]
        assert MAX_P_Q_R >= 2, "MAX_P_Q_R should be greater than or equal to 2"

        P, Q, R = self.parameter["P"], self.parameter["Q"], self.parameter["R"] = random.randint(2, MAX_P_Q_R), random.randint(2, MAX_P_Q_R), random.randint(2, MAX_P_Q_R)
        costs = self.parameter["costs"] = [[[random.randint(1, P * Q) for f in range(R)] for j in range(Q)] for i in range(P)]
        D = self.parameter["D"] = random.randint(0, R - 1)


        val = costs
        total = 0
        for k in range(R):
            for i in range(P):
                for j in range(Q):
                    total += val[i][j][k]
        # INF based on input
        INF = total + 1
        # Node indexing: S=0, for (i,j,k): id = 1 + k*(P*Q) + i*Q + j, T = 1 + (R+1)*P*Q
        node_count = 1 + (R + 1) * P * Q + 1
        S = 0
        T = node_count - 1
        # Build adjacency list
        class Edge:
            __slots__ = ('to', 'cap', 'rev')
            def __init__(self, to, cap, rev):
                self.to = to
                self.cap = cap
                self.rev = rev

        adj = [[] for _ in range(node_count)]

        def add_edge(u, v, c):
            adj[u].append(Edge(v, c, len(adj[v])))
            adj[v].append(Edge(u, 0, len(adj[u]) - 1))

        def node_id(i, j, k):
            return 1 + k * (P * Q) + i * Q + j

        # Source to layer 0 and layer edges
        for i in range(P):
            for j in range(Q):
                # Source to layer 0
                add_edge(S, node_id(i, j, 0), INF)
                # Vertical edges through layers
                for k in range(R):
                    add_edge(node_id(i, j, k), node_id(i, j, k + 1), val[i][j][k])
                # Last layer to Sink
                add_edge(node_id(i, j, R), T, INF)

        # Smoothness constraints: infinite edges for height differences > D
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i in range(P):
            for j in range(Q):
                for dx, dy in dirs:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < P and 0 <= nj < Q:
                        for k in range(D + 1, R + 2):
                            u = node_id(i, j, k - 1)
                            v = node_id(ni, nj, k - D - 1)
                            add_edge(u, v, INF)

        # Dinic's Algorithm
        level = [0] * node_count
        it = [0] * node_count

        def bfs():
            for idx in range(node_count):
                level[idx] = -1
            queue = deque([S])
            level[S] = 0
            while queue:
                u = queue.popleft()
                for e in adj[u]:
                    if e.cap > 0 and level[e.to] < 0:
                        level[e.to] = level[u] + 1
                        if e.to == T:
                            return True
                        queue.append(e.to)
            return level[T] >= 0

        def dfs(u, flow):
            if u == T:
                return flow
            for idx in range(it[u], len(adj[u])):
                e = adj[u][idx]
                if e.cap > 0 and level[u] < level[e.to]:
                    d = dfs(e.to, min(flow, e.cap))
                    if d > 0:
                        e.cap -= d
                        adj[e.to][e.rev].cap += d
                        return d
                it[u] += 1
            return 0

        flow = 0
        # Repeatedly send flow while there is a path
        while bfs():
            it = [0] * node_count
            while True:
                pushed = dfs(S, INF)
                if pushed == 0:
                    break
                flow += pushed
        assert flow > 0, "Flow should be greater than 0, indicating a valid assignment exists"
        self.parameter["gold_answer"] = flow
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            P = self.parameter["P"],
            Q = self.parameter["Q"],
            R = self.parameter["R"],
            costs = "\n".join(" ".join("c({},{},{})={}".format(i, j, f, c) for f, c in enumerate(self.parameter["costs"][i][j])) for i in range(self.parameter["P"]) for j in range(self.parameter["Q"])),
            D = self.parameter["D"],
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                matrix = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        matrix.append(list(map(int, line.split())))
                return matrix
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            assignment = processed_result
            if len(assignment) != self.parameter["P"] or any(len(row) != self.parameter["Q"] for row in assignment) :
                return self.rewards["invalid_solution"]
            
            answer, gold = 0, self.parameter["gold_answer"]
            for i in range(self.parameter["P"]) :
                for j in range(self.parameter["Q"]) :
                    if not (0 <= assignment[i][j] < self.parameter["R"]) :
                        return self.rewards["invalid_solution"]
                    for dx, dy in [(-1, 0), (+1, 0), (0, -1), (0, +1)] :
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < self.parameter["P"] and 0 <= nj < self.parameter["Q"] :
                            if abs(assignment[i][j] - assignment[ni][nj]) > self.parameter["D"] :
                                return self.rewards["invalid_solution"]
                    answer += self.parameter["costs"][i][j][assignment[i][j]]
            assert gold <= answer, "Gold answer should be less than or equal to the computed answer"

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]