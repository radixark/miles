import random
import networkx as nx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MaximumIndependentSetGrid_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2774
    prompt_template = \
r"""You are given a matrix of size {N} × {M}. Select some cells such that **no two selected cells are adjacent** (i.e., no two selected cells share a horizontal or vertical edge). Try your best to maximize the sum of the values in the selected cells. The matrix is given below (in **row-major order**):
{matrix}

**Output Format:** Output {N} lines, each with {M} digits (0 or 1) and no separators. A `1` means the corresponding cell is selected; a `0` means it is not."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the MaximumIndependentSetGrid_Environment instance.
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
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)
        NUM = self.parameter["matrix"] = [[random.randint(1, max(N, M)) for _ in range(M)] for _ in range(N)]


        # Total sum of all cell weights
        TOTAL = sum(sum(row) for row in NUM)
        # Use TOTAL as the "infinite" capacity for inter-cell edges
        INF = TOTAL
        
        # Build a directed graph for the min-cut formulation
        G = nx.DiGraph()
        SOURCE, SINK = 's', 't'
        
        # Add edges from SOURCE→odd‐parity cells and even‐parity cells→SINK
        # plus infinite‐capacity edges between adjacent cells
        for i in range(N):
            for j in range(M):
                u = (i, j)
                weight = NUM[i][j]
                
                if (i + j) % 2 == 1:
                    # Odd parity: source → u with capacity = weight
                    G.add_edge(SOURCE, u, capacity=weight)
                    # Connect to each of its neighbors with infinite capacity
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < N and 0 <= nj < M:
                            v = (ni, nj)
                            G.add_edge(u, v, capacity=INF)
                else:
                    # Even parity: u → sink with capacity = weight
                    G.add_edge(u, SINK, capacity=weight)
        
        # Compute the maximum flow (which equals the minimum cut capacity)
        flow_value, _ = nx.maximum_flow(G, SOURCE, SINK)
        
        # By König's theorem on bipartite graphs:
        # max_weight_independent_set = TOTAL - min_vertex_cover_weight
        # and min_vertex_cover_weight = flow_value
        self.parameter["gold_answer"] = TOTAL - flow_value
        assert self.parameter["gold_answer"] > 0
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            matrix = "\n".join(" ".join(str(x) for x in row) for row in self.parameter["matrix"]),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                matrix = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        matrix.append(line.strip())
                return matrix
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            N, M = self.parameter["N"], self.parameter["M"]
            solution = processed_result
            
            if len(solution) != N or any(len(row) != M for row in solution) :
                return self.rewards["wrong_format"]
            if any(c not in '01' for row in solution for c in row) :
                return self.rewards["wrong_format"]
            
            answer, gold = 0, self.parameter["gold_answer"]
            for i in range(N) :
                for j in range(M) :
                    if solution[i][j] == '1' :
                        answer += self.parameter["matrix"][i][j]
                        for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)) :
                            ni, nj = i + di, j + dj
                            if 0 <= ni < N and 0 <= nj < M and solution[ni][nj] == '1' :
                                return self.rewards["invalid_solution"]
            assert answer <= gold, "Answer should not exceed the gold answer"

            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / self.parameter["gold_answer"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == self.parameter["gold_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]