import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class FractionalProgramming_BipartiteGraphMatching_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3705
    prompt_template = \
r"""You are given two matrices A and B of size {N} × {N} (0-indexed).

Matrix A is (given in **row-major order**, with each row represented as a list of integers separated by spaces):
{A}

Matrix B is (given in **row-major order**, with each row represented as a list of integers separated by spaces):
{B}

Please find a permutation P of indices from 0 to {N_minus_1}, i.e., P[0], P[1], ..., P[{N_minus_1}], such that the following value is maximized: (A[0][P[0]] + A[1][P[1]] + ... + A[{N_minus_1}][P[{N_minus_1}]]) / (B[0][P[0]] + B[1][P[1]] + ... + B[{N_minus_1}][P[{N_minus_1}]])

**Output Format:** A single line containing P[0], P[1], ..., P[{N_minus_1}], separated by spaces."""

    def __init__(self,
                 max_proportion : int = 2,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the FractionalProgramming_BipartiteGraphMatching_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_proportion = max_proportion
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


        B = self.parameter["B"] = [[random.randint(1, N) for _ in range(N)] for _ in range(N)]
        A = self.parameter["A"] = [[random.randint(1, self.max_proportion * b) for b in B_row] for B_row in B]


        def max_weight_matching_networkx(W):
            # Create bipartite graph
            G = networkx.Graph()
            N = len(W)
            
            # Add nodes (left nodes: 0 to N-1, right nodes: N to 2N-1)
            left_nodes = list(range(N))
            right_nodes = list(range(N, 2*N))
            G.add_nodes_from(left_nodes, bipartite=0)
            G.add_nodes_from(right_nodes, bipartite=1)
            
            # Add weighted edges
            for i in range(N):
                for j in range(N):
                    G.add_edge(i, N + j, weight=W[i][j])
            
            # Find maximum weight matching
            matching = networkx.max_weight_matching(G, maxcardinality=True)
            
            # Convert to P array format
            P = [-1] * N
            for edge in matching:
                left, right = edge
                if left < N:  # left is from left side
                    P[left] = right - N
                else:  # left is actually from right side
                    P[right] = left - N
            
            total_weight = sum(W[i][P[i]] for i in range(N) if P[i] != -1)
            
            return P, total_weight

        l, r = 0.0, max(max(a / b for a, b in zip(A_row, B_row)) for A_row, B_row in zip(A, B))
        P = None
        for _ in range(256) :
            mid = (l + r) / 2
            W = [[A[i][j] - mid * B[i][j] for j in range(N)] for i in range(N)]
            
            tempP, total_weight = max_weight_matching_networkx(W)
            
            if total_weight >= 0 :
                l = mid
                P = tempP.copy()
            else:
                r = mid
        self.parameter["reference_answer"] = " ".join(map(str, P))

        self.parameter["gold_SumA"], self.parameter["gold_SumB"] = sum(A[i][P[i]] for i in range(N)), sum(B[i][P[i]] for i in range(N))
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            A = "\n".join(" ".join(map(str, row)) for row in self.parameter["A"]),
            B = "\n".join(" ".join(map(str, row)) for row in self.parameter["B"]),
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

            P = processed_result
            if len(P) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if set(P) != set(range(self.parameter["N"])) :
                return self.rewards["invalid_solution"]

            answer_SumA, answer_SumB = sum(self.parameter["A"][i][P[i]] for i in range(self.parameter["N"])), sum(self.parameter["B"][i][P[i]] for i in range(self.parameter["N"]))
            gold_SumA, gold_SumB = self.parameter["gold_SumA"], self.parameter["gold_SumB"]
            # gold_SumA / gold_SumB >= answer_SumA / answer_SumB   <=>   gold_SumA * answer_SumB >= answer_SumA * gold_SumB
            assert gold_SumA * answer_SumB >= answer_SumA * gold_SumB, "gold_SumA * answer_SumB should be greater than or equal to answer_SumA * gold_SumB"

            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                # (answer_SumA / answer_SumB) / (gold_SumA / gold_SumB) = (answer_SumA * gold_SumB) / (answer_SumB * gold_SumA)
                return self.rewards["rewarding_weight"] * (((answer_SumA * gold_SumB) / (answer_SumB * gold_SumA)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * ((answer_SumA * gold_SumB) == (answer_SumB * gold_SumA))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]