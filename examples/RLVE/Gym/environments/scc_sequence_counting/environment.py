import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class SCC_Sequence_Counting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P5241
    prompt_template = \
r"""Consider a directed graph with {N} vertices, initially with no edges. You may choose an arbitrary list of **E directed edges** to add to the graph, under the following constraints:
- Each edge connects two **distinct** vertices (i.e., no self-loops).
- No two edges in the list are the same.
- The edges are added **one by one** in the given order of the list.

After adding each edge, compute the number of **strongly connected components (SCCs)** in the current graph (with the edges added so far) and record it; this produces a sequence of E integers — we call this an **SCC sequence**. Your task is to compute, for each possible value of E from 1 to {N} × ({N} - 1), how many **distinct SCC sequences** can be produced.

Output {N} × ({N} - 1) integers in one line, separated by spaces. The i-th number (1 ≤ i ≤ {N} × ({N} - 1)) is the number of distinct SCC sequences that can be obtained when E = i, **modulo {MOD}**."""

    def __init__(self,
                 max_MOD : int = 1000000,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the SCC_Sequence_Counting_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_MOD = max_MOD
        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)


        # Precompute the “p_limit” array
        p_limit = [0] * (N + 1)
        for i in range(1, N + 1):
            # same formula as (n - i + 1)*(n - 1) + (i - 1)*(i - 2)/2
            p_limit[i] = (N - i + 1) * (N - 1) + (i - 1) * (i - 2) // 2

        # f and sf are 2×(N+2)×(N+2) to allow indexing up to [j+1][k-1]
        f  = [[[0] * (N + 2) for _ in range(N + 2)] for _ in range(2)]
        sf = [[[0] * (N + 2) for _ in range(N + 2)] for _ in range(2)]

        # g and sg are 2×(N+2)
        g  = [[0] * (N + 2) for _ in range(2)]
        sg = [[0] * (N + 2) for _ in range(2)]

        # ans[E] will hold the answer for sequence‐length E
        ans = [0] * (N * (N - 1) + 2)

        # --- initialize for E = 1 ---
        f[1][N][1] = 1
        ans[1] = 1
        for i in range(1, N + 1):
            sf[1][i][1] = 1

        # --- first phase: E = 2 … min(N*(N-1), 2*N) ---
        maxE = min(N * (N - 1), N << 1)
        for E in range(2, maxE + 1):
            op   = E & 1
            prev = op ^ 1

            # zero out f[op]
            for j in range(1, N + 1):
                for k in range(1, N + 1):
                    f[op][j][k] = 0

            # DP recurrence
            for j in range(1, N + 1):
                if E <= p_limit[j]:
                    for k in range(1, N + 1):
                        # only valid if E + j >= N + k - 1
                        if E + j >= N + k - 1:
                            f[op][j][k] = (f[prev][j][k] + sf[prev][j + 1][k - 1]) % MOD

            # build sf[op] and accumulate ans[E]
            total = 0
            for j in range(N, 0, -1):
                for k in range(1, N + 1):
                    sf[op][j][k] = (sf[op][j + 1][k] + f[op][j][k]) % MOD
                    total = (total + f[op][j][k]) % MOD
            ans[E] = total

        # --- prepare g[0] and sg[0] from f[0] ---
        for j in range(1, N + 1):
            s = 0
            for k in range(1, N + 1):
                s = (s + f[0][j][k]) % MOD
            g[0][j] = s

        for j in range(N, 0, -1):
            sg[0][j] = (sg[0][j + 1] + g[0][j]) % MOD

        # --- second phase: E = 2*N+1 … N*(N-1) ---
        for E in range((N << 1) + 1, N * (N - 1) + 1):
            op   = E & 1
            prev = op ^ 1

            # zero out g[op]
            for j in range(1, N + 1):
                g[op][j] = 0

            # recurrence for g
            for j in range(1, N + 1):
                if E <= p_limit[j]:
                    g[op][j] = sg[prev][j]

            # build sg[op] and accumulate ans[E]
            total = 0
            for j in range(N, 0, -1):
                sg[op][j] = (sg[op][j + 1] + g[op][j]) % MOD
                total = (total + g[op][j]) % MOD
            ans[E] = total

        # output ans[1..N*(N-1)]
        self.parameter["gold_answer"] = ans[1 : N * (N - 1) + 1]
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["gold_answer"]))


    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], MOD = self.parameter["MOD"])


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

            if len(processed_result) != self.parameter["N"] * (self.parameter["N"] - 1) :
                return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["gold_answer"], processed_result)) / (self.parameter["N"] * (self.parameter["N"] - 1))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]