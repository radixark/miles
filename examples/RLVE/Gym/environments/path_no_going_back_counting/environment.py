import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Path_NoGoingBack_Counting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2151
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices labeled from `0` to `{N_minus_1}`. The graph contains the following undirected edges (no repeated edges):  
{edges}

Please count the number of paths from vertex `0` to vertex `{N_minus_1}` that satisfy the following conditions:
- The path has exactly {T} edges.
- You may not immediately return to the previous vertex. That is, if you move along edge `(u, v)` from `u` to `v`, you cannot move back to `u` in the very next step.

**Output Format:** Your final answer should be a single integer — the number of valid paths, modulo {MOD}."""
    MOD = 10000

    def __init__(self,
                 wrong_format: float = -1.0, wrong_range: float = -0.5, correct_answer: float = +1.0, wrong_answer: float = 0.0,
                 **kwargs):
        """
        Initialize the Path_NoGoingBack_Counting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "wrong_range": wrong_range,
            "correct_answer": correct_answer,
            "wrong_answer": wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_M" in self.parameter, "MAX_M must be set in the parameter"
        MAX_M = self.parameter["MAX_M"]
        assert MAX_M >= 3, "MAX_M must be at least 3"
        
        M = self.parameter["M"] = random.randint(3, MAX_M)

        valid_N = [N for N in range(3, (M + 1) + 1) if M <= N * (N - 1) // 2]
        N = self.parameter["N"] = random.choice(valid_N)
        assert N - 1 <= M <= N * (N - 1) // 2, "M must be at least N - 1 and at most N * (N - 1) / 2"

        T = self.parameter["T"] = random.randint(1, 2 ** N)

        edges = self.parameter["edges"] = []
        initial_permutation = list(range(N))
        random.shuffle(initial_permutation)
        for u, v in zip(initial_permutation, initial_permutation[1 :]):
            edges.append((min(u, v), max(u, v)))
        if len(edges) < M :
            edges += random.sample(list(set((u, v) for u in range(N) for v in range(u + 1, N)) - set(edges)), M - len(edges))
        random.shuffle(edges)
        
        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"


        Start, End = 0, N - 1

        x = [-1]   # x[i] = source vertex of the i-th “edge”
        y = [Start]    # y[i] = destination vertex of the i-th “edge”
        for u, v in edges:
            x.append(u); y.append(v)
            x.append(v); y.append(u)

        cnt = len(x)

        # Precompute reversal-pair for each directed edge
        pair = [-1] * cnt
        for j in range(1, cnt):
            if j % 2 == 1:
                pair[j] = j + 1
            else:
                pair[j] = j - 1

        # Build the adjacency matrix A of the “edge-graph”
        A = [[0] * cnt for _ in range(cnt)]
        for i in range(cnt):
            yi = y[i]
            Ai = A[i]
            for j in range(cnt):
                if yi == x[j] and i != j and i != pair[j]:
                    Ai[j] = 1

        # Matrix multiplication (MODular)
        def mat_mult(A, B):
            n = len(A)
            C = [[0] * n for _ in range(n)]
            for i in range(n):
                Ai = A[i]
                Ci = C[i]
                for k in range(n):
                    if Ai[k]:
                        aik = Ai[k]
                        Bk = B[k]
                        for j in range(n):
                            Ci[j] = (Ci[j] + aik * Bk[j]) % self.MOD
            return C

        # Fast exponentiation of matrix A^power
        def mat_pow(mat, power):
            n = len(mat)
            # identity
            res = [[0] * n for _ in range(n)]
            for i in range(n):
                res[i][i] = 1
            while power:
                if power & 1:
                    res = mat_mult(res, mat)
                mat = mat_mult(mat, mat)
                power >>= 1
            return res

        # Compute A^T
        A_exp = mat_pow(A, T)

        # The number of walks of length T from S to T is the sum over all
        # directed edges i ending at vertex T of (A^T)[0][i]
        ans = 0
        row0 = A_exp[0]
        for i in range(cnt):
            if y[i] == End:
                ans = (ans + row0[i]) % self.MOD

        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
            T = self.parameter["T"],
            MOD = self.MOD,
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
            if not (0 <= processed_result < self.MOD) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]