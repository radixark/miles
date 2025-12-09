import random
from typing import Optional, List, Tuple
from Gym.environment import VerifiableEnvironment


class Minimum_DominatingSet_Grid_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3888
    prompt_template = \
r"""We have a grid with {N} rows and {M} columns (1-based indices). The cost of cell (i, j) is F[i][j]:
{F}

Select a set of **distinct** cells S such that every cell is either in S or has at least one **orthogonally adjacent** selected neighbor (up, down, left, or right). Minimize the total cost of selected cells (i.e., the sum of F[i][j] for all (i,j) ∈ S). Output K (the number of selected cells) lines: each line contains two integers `i j` (1-based), the row and column of a selected cell (in any order)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Minimum_DominatingSet_Grid_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "unsuccessful_solution" : unsuccessful_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N = self.parameter["N"] = random.randint(2, MAX_N_M)
        M = self.parameter["M"] = random.randint(2, MAX_N_M)

        F = self.parameter["F"] = [[random.randint(1, N * M) for j in range(M)] for i in range(N)]


        S = 1 << M
        ALL = S - 1

        # --- Precompute helpers ---
        # popcount for every mask
        ones = [0] * S
        for m in range(S):
            ones[m] = m.bit_count()

        # shift coverage within a row (same row left/right neighbors)
        shift_cov = [0] * S
        for m in range(S):
            shift_cov[m] = (m | ((m << 1) & ALL) | (m >> 1)) & ALL

        # map bit -> column index
        bit_to_idx = {}
        for c in range(M):
            bit_to_idx[1 << c] = c

        # row_sums[i][mask]: cost of choosing 'mask' on row i (1-based rows for DP)
        # add a dummy row N+1 with all zero costs (to flush coverage of the last real row)
        row_sums = [[0] * S for _ in range(N + 2)]  # index 1..N, N+1 is zeros

        for i in range(1, N + 1):
            costs = F[i - 1]
            rs = row_sums[i]
            for mask in range(S):
                total = 0
                x = mask
                while x:
                    t = x & -x
                    total += costs[bit_to_idx[t]]
                    x -= t
                rs[mask] = total
        # row_sums[N+1] already zero

        # supersets list: for each 'need' mask, all p where p is a superset of 'need'
        supersets = [[] for _ in range(S)]
        for need in range(S):
            rem = ALL ^ need  # bits we are free to choose
            x = rem
            while True:
                supersets[need].append(need | x)
                if x == 0:
                    break
                x = (x - 1) & rem

        INF = float('inf')

        # DP arrays: f[p][j] and g[p][j]
        # f: minimal cost; g: number of depots (tie-breaker)
        f = [[INF] * S for _ in range(S)]
        g = [[INF] * S for _ in range(S)]

        # Initialize for first row: previous row (k) is 0
        rs1 = row_sums[1]
        for j in range(S):
            f[j][0] = rs1[j]
            g[j][0] = ones[j]

        # Transition rows 2..N+1 (N+1 is dummy zero-cost row)
        for i in range(2, N + 2):
            nf = [[INF] * S for _ in range(S)]
            ng = [[INF] * S for _ in range(S)]
            rsi = row_sums[i]

            for j in range(S):            # mask for row i-1
                sj = shift_cov[j]
                fj = f[j]
                gj = g[j]
                for k in range(S):        # mask for row i-2
                    base_cost = fj[k]
                    if base_cost == INF:
                        continue
                    base_cnt = gj[k]
                    need = ALL ^ (sj | k)   # columns still needing coverage on row i-1
                    for p in supersets[need]:  # mask for row i
                        v = base_cost + rsi[p]
                        c = base_cnt + ones[p]
                        if v < nf[p][j]:
                            nf[p][j] = v
                            ng[p][j] = c
                        elif v == nf[p][j] and c < ng[p][j]:
                            ng[p][j] = c

            f, g = nf, ng

        # Finalize: last (dummy) row must be p=0; scan any j
        best_cost = INF
        best_cnt = INF
        f0 = f[0]
        g0 = g[0]
        for j in range(S):
            v = f0[j]
            if v < best_cost:
                best_cost = v
                best_cnt = g0[j]
            elif v == best_cost and g0[j] < best_cnt:
                best_cnt = g0[j]

        assert best_cost > 0, "gold_answer must be greater than 0"
        self.parameter["gold_answer"] = best_cost
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            F = "\n".join(" ".join("F[{}][{}]={}".format(i, j, Fij) for j, Fij in enumerate(Fi, start = 1)) for i, Fi in enumerate(self.parameter["F"], start = 1)),
        )


    def _process(self, answer : Optional[str]) -> Optional[List[Tuple[int, int]]] :
        if answer is not None :
            answer = answer.strip()
            try :
                cells = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        i, j = map(int, line.split())
                        cells.append((i, j))
                return cells
            except :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            selected = [[False] * self.parameter["M"] for i in range(self.parameter["N"])]
            for i, j in processed_result :
                if not (1 <= i <= self.parameter["N"] and 1 <= j <= self.parameter["M"]) :
                    return self.rewards["invalid_solution"]
                if selected[i - 1][j - 1] :
                    return self.rewards["invalid_solution"]
                selected[i - 1][j - 1] = True
            
            dxs = [0, 0, 0, -1, +1]
            dys = [0, -1, +1, 0, 0]
            for i in range(self.parameter["N"]) :
                for j in range(self.parameter["M"]) :
                    if not any(0 <= i + dx < self.parameter["N"] and 0 <= j + dy < self.parameter["M"] and selected[i + dx][j + dy] for dx, dy in zip(dxs, dys)) :
                        return self.rewards["unsuccessful_solution"]
            
            answer, gold = sum(self.parameter["F"][i - 1][j - 1] for i, j in processed_result), self.parameter["gold_answer"]
            assert 0 < gold <= answer, "gold_answer must be greater than 0 and less than or equal to answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]