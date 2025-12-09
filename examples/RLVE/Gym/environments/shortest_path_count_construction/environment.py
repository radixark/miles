import random
from typing import Optional
from Gym.environment import VerifiableEnvironment
import numpy as np
from typing import Tuple

class ShortestPathCountConstruction_Environment(VerifiableEnvironment) : # Source: https://codeforces.com/problemset/problem/388/B
    prompt_template = \
r"""Please construct a simple undirected graph with N vertices, such that the number of shortest paths between vertex 1 and vertex 2 is {K}. Since there are multiple valid graphs satisfying the condition, you can output any of them.
{N_constraint}

Please strictly follow the output format without additional stuff:
1. The first line must contain an integer N.
2. The next N lines each contain a string of length N, representing the adjacency matrix G with N rows and N columns. Each element of the matrix must be 'N' or 'Y'. If Gij is 'Y', then graph G has a edge connecting vertex i and vertex j. Consider the graph vertexes are numbered from 1 to N. The graph must be undirected and simple: Gii = 'N' and Gij = Gji must hold. And there must be at least one path between vertex 1 and vertex 2.
"""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0, trivial_solution_penalty : float = -0.5,
                 **kwargs) :
        """
        Initialize the ShortestPathCountConstruction instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
            "trivial_solution_penalty" : trivial_solution_penalty,
        }
    
    def _generate(self) -> None :
        assert "MAX_K" in self.parameter, "MAX_K is required in parameter"
        MAX_K = self.parameter["MAX_K"]
        assert MAX_K >= 3, "MAX_K should be greater than or equal to 3"
        
        K = self.parameter["K"] = random.randint(3, MAX_K)

        if K >= 12 :
            self.parameter["N_constraint"] = min(((len(bin(K)[2 :]) * 3 + 1) + 1) * 2, K + 2)
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            K = self.parameter["K"],
            N_constraint = "Please ensure that the number of verticies N is fewer than {}.".format(self.parameter["N_constraint"]) if self.parameter["K"] >= 12 else "Please try your best to avoid constructing a trivial solution with N = {K} + 2 (by just putting {K} intermediate vertices between vertex 1 and vertex 2).".format(K = self.parameter["K"])
        )

    def _process(self, answer : Optional[str]) -> Optional[Tuple[int, np.ndarray]] :
        if answer is not None :
            try :
                answer = answer.strip()
                N = int(answer[:answer.find("\n")])
                answer = answer[answer.find("\n") + 1 :]
                assert(sum([1 if c=='\n' else 0 for c in answer]) == N - 1)
                answer = answer.splitlines()
                adjacency_matrix = np.ndarray((N, N), dtype=int)
                for i in range(N) :
                    assert(len(answer[i]) == N)
                    for j in range(N) :
                        # check if the adjacency matrix is valid: ('N' or 'Y')
                        assert answer[i][j] in ['N', 'Y']
                        adjacency_matrix[i, j] = answer[i][j] == 'Y'
                # check if the adjacency matrix is valid: (symmetric, no self-loops)
                assert(np.all(adjacency_matrix == adjacency_matrix.T))
                assert(np.all(np.diag(adjacency_matrix) == 0))
                return N, adjacency_matrix
            except (ValueError, AssertionError) :
                return None
        else :
            return None
    
    def count_shortest_paths(self, N, adjacency_matrix):
        """
        Assume the format is completely correct.
        Count the number of shortest paths between vertex 1 and vertex 2 in the given graph.
        Use matrix multiplication instead of BFS, since numpy is faster than python for loop.
        """

        start_node_idx = 0
        end_node_idx = 1

        current_paths_vec = np.zeros(N, dtype=int)
        current_paths_vec[start_node_idx] = 1

        # enumerate the shortest path length
        for k in range(1, N):
            next_paths_vec = adjacency_matrix @ current_paths_vec
            
            # check if there is a path to the end node
            if next_paths_vec[end_node_idx] > 0:
                return next_paths_vec[end_node_idx]
            
            # update the vector for the next iteration
            current_paths_vec = next_paths_vec

        # if the loop ends without finding a path, then there is no path from the start node to the end node
        return 0

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            N, adjacency_matrix = processed_result
            if N < 2 :
                return self.rewards["wrong_format"]
            real_K = int(self.count_shortest_paths(N, adjacency_matrix))
            if self.parameter["K"] >= 12 and N >= self.parameter["N_constraint"] : # a trivial solution is N = K+2, and we try to avoid it by penalizing a big N (when k>=12, 3\lceil\log k\rceil+1 < k+2)
                return self.rewards["trivial_solution_penalty"]
            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                return self.rewards["rewarding_weight"] * ((min(real_K, self.parameter["K"]) / max(real_K, self.parameter["K"])) ** self.rewards["rewarding_beta"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]