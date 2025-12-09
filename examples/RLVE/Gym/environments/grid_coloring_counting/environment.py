import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class GridColoringCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3158
    prompt_template = \
r"""You are given a grid of size {N} × {M}. You may color some cells (and leave others uncolored) using {C} colors labeled from 0 to {C_minus_1}, such that:
1. No two different colors appear in the same row or the same column.
2. Color `i` is used exactly X[i] times. The array X is given as: {Xs}

Please compute the number of valid colorings modulo {MOD}."""

    def __init__(self,
                 max_MOD : int = 10000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the GridColoringCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_MOD = max_MOD
        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        while True :
            N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)
            sum_X = random.randint(1, N * M)
            C = self.parameter["C"] = random.randint(1, min(N, M, sum_X))

            deltas = random.sample(range(1, sum_X), C - 1)
            deltas.sort()
            deltas = [0] + deltas + [sum_X]
            self.parameter["Xs"] = Xs = [deltas[i + 1] - deltas[i] for i in range(C)]
            assert len(Xs) == C and all(x > 0 for x in Xs), "Xs should be a non-empty list of positive integers"

            MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)


            # Precompute binomial coefficients up to N*M
            total_cells = N * M
            comb = [[0] * (total_cells + 1) for _ in range(total_cells + 1)]
            for i in range(total_cells + 1):
                comb[i][0] = 1
                for j in range(1, i + 1):
                    comb[i][j] = (comb[i - 1][j] + comb[i - 1][j - 1]) % MOD

            # f[i][j][k]: number of ways to place first k colors into an i×j subboard
            f = [[[0] * (C + 1) for _ in range(M + 1)] for __ in range(N + 1)]
            f[0][0][0] = 1

            # Process each color one by one
            for k in range(1, C + 1):
                x = Xs[k - 1]
                # g[a][b]: number of ways to place x pieces of this color into an a×b rectangle
                # so that every row and column used by it has at least one piece,
                # by inclusion–exclusion
                g = [[0] * (M + 1) for _ in range(N + 1)]
                for a in range(1, N + 1):
                    for b in range(1, M + 1):
                        if a * b < x:
                            continue
                        # total ways to choose x squares out of a*b
                        val = comb[a * b][x]
                        # subtract configurations that leave an unused border row or column
                        for la in range(1, a + 1):
                            for lb in range(1, b + 1):
                                if la < a or lb < b:
                                    val -= g[la][lb] * comb[a][la] * comb[b][lb]
                        g[a][b] = val % MOD

                # Transition: add this color's placements to all previous subboards
                for i in range(1, N + 1):
                    for j in range(1, M + 1):
                        # split the i×j board into an l×r part (already filled with k−1 colors)
                        # and a (i−l)×(j−r) part filled with k-th color
                        for l in range(i):
                            for r in range(j):
                                ti, tj = i - l, j - r
                                if ti * tj < x:
                                    continue
                                ways = (
                                    f[l][r][k - 1]
                                    * g[ti][tj]
                                    * comb[N - l][ti]
                                    * comb[M - r][tj]
                                ) % MOD
                                f[i][j][k] = (f[i][j][k] + ways) % MOD

            # Sum over all non-empty subboards
            answer = 0
            for i in range(1, N + 1):
                for j in range(1, M + 1):
                    answer = (answer + f[i][j][C]) % MOD

            if answer > 0 :
                self.parameter["reference_answer"] = answer
                break
    

    def _prompt_generate(self) -> str :
        C = self.parameter["C"]
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            C = C,
            C_minus_1 = C - 1,
            Xs = " ".join("X[{}]={}".format(i, x) for i, x in enumerate(self.parameter["Xs"])),
            MOD = self.parameter["MOD"],
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
            if not (0 <= processed_result < self.parameter["MOD"]) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]