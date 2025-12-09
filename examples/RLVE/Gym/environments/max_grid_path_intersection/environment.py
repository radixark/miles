import random
from typing import Optional
from collections import deque
from Gym.environment import VerifiableEnvironment


class MaxGridPathIntersection_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2045
    prompt_template = \
r"""You are given an {N} × {N} grid (0-indexed) of non-negative integers (given in **row-major order**):
{grid}

You will start at cell (0, 0) and move to cell ({N_minus_1}, {N_minus_1}) exactly {K} times. Each time, you can only move **right** or **down** at each step. When you step on a cell during a path, you collect its value and set it to 0 (so future paths will see it as 0). Your goal is to **maximize the total sum** collected across all {K} paths.

**Output Format:** A single integer — the maximum total sum that can be collected after {K} such paths."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MaxGridPathIntersection_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        K = self.parameter["K"] = random.randint(1, N // 2)

        A = self.parameter["grid"] = [[random.randint(0, N) for _ in range(N)] for _ in range(N)]

        
        def max_cost_flow(N, K, A):
            # Number of nodes: each cell has in-node and out-node
            total_nodes = 2 * N * N
            # Adjacency list: each entry is [to, capacity, cost, rev]
            ADJ = [[] for _ in range(total_nodes)]

            def add_edge(u, v, cap, cost):
                # forward edge
                forward = [v, cap, cost, None]
                # reverse edge
                backward = [u, 0, -cost, None]
                # link edges for capacity updates
                forward[3] = backward
                backward[3] = forward
                ADJ[u].append(forward)
                ADJ[v].append(backward)

            def node_id(i, j, is_out):
                # 0-indexed: cells at (i, j) share indices 0..N*N-1 for in-nodes,
                # N*N..2*N*N-1 for out-nodes
                base = N * N if is_out else 0
                return base + i * N + j

            # Build the flow network
            for i in range(N):
                for j in range(N):
                    in_id = node_id(i, j, False)
                    out_id = node_id(i, j, True)
                    # Pick the cell's value on one of the K visits
                    add_edge(in_id, out_id, 1, A[i][j])    # one with reward
                    add_edge(in_id, out_id, K - 1, 0)      # others free
                    # Move right or down (up to K walkers)
                    if j + 1 < N:
                        add_edge(out_id, node_id(i, j + 1, False), K, 0)
                    if i + 1 < N:
                        add_edge(out_id, node_id(i + 1, j, False), K, 0)

            s = node_id(0, 0, False)
            t = node_id(N - 1, N - 1, True)
            total_cost = 0

            # If K is zero, there is no flow and cost is zero
            if K == 0:
                return 0

            # Successive SPFA for maximum-cost flow
            while True:
                DIST = [float('-inf')] * total_nodes
                FLOW = [0] * total_nodes
                INQUEUE = [False] * total_nodes
                PREV_NODE = [None] * total_nodes
                PREV_EDGE = [None] * total_nodes

                queue = deque([s])
                DIST[s] = 0
                FLOW[s] = K   # maximum possible augment per iteration
                INQUEUE[s] = True

                # Find longest path from s to t in residual graph
                while queue:
                    u = queue.popleft()
                    INQUEUE[u] = False
                    for edge in ADJ[u]:
                        v, cap, cost, rev = edge
                        if cap > 0 and DIST[v] < DIST[u] + cost:
                            DIST[v] = DIST[u] + cost
                            FLOW[v] = min(FLOW[u], cap)
                            PREV_NODE[v] = u
                            PREV_EDGE[v] = edge
                            if not INQUEUE[v]:
                                queue.append(v)
                                INQUEUE[v] = True

                # If there's no augmenting path, we're done
                if DIST[t] == float('-inf'):
                    break

                # Augment along the path
                f = FLOW[t]
                total_cost += f * DIST[t]
                v = t
                while v != s:
                    edge = PREV_EDGE[v]
                    # reduce forward capacity
                    edge[1] -= f
                    # increase reverse capacity
                    edge[3][1] += f
                    v = PREV_NODE[v]

            return total_cost
        
        self.parameter["reference_answer"] = max_cost_flow(N, K, A)
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            K = self.parameter["K"],
            grid = "\n".join(" ".join(map(str, row)) for row in self.parameter["grid"]),
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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]