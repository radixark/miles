import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinPairSumMultiplicationPermutation_Environment(VerifiableEnvironment) : # Submitted to https://www.luogu.com.cn/problem/P3236
    prompt_template = \
r"""You are given two matrices `A` and `B`, each of size {N} × {N}:
{matrix_A}
{matrix_B}

You need to find a permutation P of indices from 0 to {N_minus_1} such that the value (sum of A[0][P[0]], A[1][P[1]], ..., A[{N_minus_1}][P[{N_minus_1}]]) multiplied by (sum of B[0][P[0]], B[1][P[1]], ..., B[{N_minus_1}][P[{N_minus_1}]]) is minimized.

**Output Format:** A single line containing P[0], P[1], ..., P[{N_minus_1}], separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinPairSumMultiplicationPermutation_Environment instance.
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

        A = self.parameter["A"] = [[random.randint(1, N) for _ in range(N)] for _ in range(N)]
        B = self.parameter["B"] = [[random.randint(1, N) for _ in range(N)] for _ in range(N)]


        def hungarian(CX: int, CY: int, A, B, N, BIG):
            """
            Minimise   Σ ( A[i][j]*CX + B[i][j]*CY ),  i,j a permutation.
            Returns the permutation as a list row_match[i] = chosen column.
            """
            U = [0] * (N + 1)
            V = [0] * (N + 1)
            P = [0] * (N + 1)
            WAY = [0] * (N + 1)

            for i in range(1, N + 1):                   # rows 1..N
                P[0] = i
                j0 = 0
                MINV = [BIG] * (N + 1)
                USED = [False] * (N + 1)
                USED[0] = True
                while True:
                    USED[j0] = True
                    i0 = P[j0]
                    delta = BIG
                    j1 = 0
                    for j in range(1, N + 1):
                        if not USED[j]:
                            cur = (A[i0 - 1][j - 1] * CX + B[i0 - 1][j - 1] * CY) - U[i0] - V[j]
                            if cur < MINV[j]:
                                MINV[j] = cur
                                WAY[j] = j0
                            if MINV[j] < delta:
                                delta = MINV[j]
                                j1 = j
                    for j in range(N + 1):              # shift potentials
                        if USED[j]:
                            U[P[j]] += delta
                            V[j] -= delta
                        else:
                            MINV[j] -= delta
                    j0 = j1
                    if P[j0] == 0:
                        break                           # free column found
                # -------   augment along the path   -------
                while True:
                    j1 = WAY[j0]
                    P[j0] = P[j1]
                    j0 = j1
                    if j0 == 0:
                        break

            row_match = [-1] * N
            for j in range(1, N + 1):
                if P[j] != 0:
                    row_match[P[j] - 1] = j - 1
            return row_match


        # ----------   tiny Point helper   ----------
        class Point:
            __slots__ = ("x", "y")

            def __init__(self, x=0, y=0):
                self.x = x
                self.y = y

            def calc(self, A, B):                       # ⟨self ,  (A.y-B.y , B.x-A.x)⟩
                return self.x * (A.y - B.y) + self.y * (B.x - A.x)

        # ----------   solve one test case   ----------
        def solve_case():
            # -------- derive a SAFE 'BIG' sentinel for this test case ----------
            MAX_A = max(max(row) for row in A)
            MAX_B = max(max(row) for row in B)
            # every CX or CY equals a difference of two sums of ≤ N*MAX_A / B
            SUM_BOUND = N * max(MAX_A, MAX_B)          # ≤ 14 000 with constraints
            BIG = (MAX_A + MAX_B) * SUM_BOUND + 1      # > any possible edge cost

            # -------  closure:  run Hungarian, return Point(sumA,sumB)  -------
            def MM(cx: int, cy: int) -> Point:
                match = hungarian(cx, cy, A, B, N, BIG)
                sx = sy = 0
                for i in range(N):
                    j = match[i]
                    sx += A[i][j]
                    sy += B[i][j]
                return Point(sx, sy)

            POINT_A = MM(1, 0)          # minimal ΣA
            POINT_B = MM(0, 1)          # minimal ΣB
            best = min(POINT_A.x * POINT_A.y, POINT_B.x * POINT_B.y)

            # -------  recursively walk the lower convex hull  -------
            def recurse(P: Point, Q: Point):
                nonlocal best
                C = MM(P.y - Q.y, Q.x - P.x)
                best = min(best, C.x * C.y)
                if C.calc(P, Q) >= P.calc(P, Q):        # C lies on / below PQ
                    return
                recurse(P, C)
                recurse(C, Q)

            recurse(POINT_A, POINT_B)
            return best
        self.parameter["gold_answer"] = solve_case()
        assert self.parameter["gold_answer"] > 0
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            matrix_A = "\n".join(" ".join("A[{}][{}]={}".format(i, j, self.parameter["A"][i][j]) for j in range(N)) for i in range(N)),
            matrix_B = "\n".join(" ".join("B[{}][{}]={}".format(i, j, self.parameter["B"][i][j]) for j in range(N)) for i in range(N)),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            P = processed_result
            if len(P) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if set(P) != set(range(self.parameter["N"])) :
                return self.rewards["invalid_solution"]
            
            answer, gold = sum(self.parameter["A"][i][P[i]] for i in range(self.parameter["N"])) * sum(self.parameter["B"][i][P[i]] for i in range(self.parameter["N"])), self.parameter["gold_answer"]
            assert gold <= answer, "The answer should be greater than or equal to the gold answer"

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]