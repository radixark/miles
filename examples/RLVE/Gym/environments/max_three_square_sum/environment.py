import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MaxThreeSquareSum_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3625
    prompt_template = \
r"""You are given a grid of size {N} × {M}, where each cell contains an integer. Please find three **non-overlapping** {K} × {K} squares in the grid such that the sum of all values in the three squares is maximized. The grid is provided as follows:
{grid}

**Output Format:** Output a single integer — the maximum possible sum of values from the three non-overlapping {K} × {K} squares."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MaxThreeSquareSum_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 4, "MAX_N_M should be greater than or equal to 4"

        N = self.parameter["N"] = random.randint(4, MAX_N_M)
        M = self.parameter["M"] = random.randint(4, MAX_N_M)
        K = self.parameter["K"] = random.randint(2, min(N, M) // 2)
        A = self.parameter["A"] = [[random.randint(0, MAX_N_M) for _ in range(M)] for _ in range(N)]


        S = [[0]*(M+1) for _ in range(N+1)]
        for i in range(N):
            for j in range(M):
                S[i+1][j+1] = A[i][j] + S[i][j+1] + S[i+1][j] - S[i][j]

        # cal(i,j) = sum of K×K ending at (i,j)
        def cal(i, j):
            if i < K-1 or j < K-1:
                return 0
            return (S[i+1][j+1]
                - S[i+1-K][j+1]
                - S[i+1][j+1-K]
                + S[i+1-K][j+1-K])

        # mxx[i] = best K×K whose bottom row is i
        # mxy[j] = best K×K whose right  col is j
        mxx = [0]*N
        mxy = [0]*M
        for i in range(K-1, N):
            for j in range(K-1, M):
                v = cal(i, j)
                if v > mxx[i]: mxx[i] = v
                if v > mxy[j]: mxy[j] = v

        # a[l][r] = max(mxx[t] for t in [l..r])
        a = [[0]*N for _ in range(N)]
        for l in range(N):
            a[l][l] = mxx[l]
            for r in range(l+1, N):
                a[l][r] = max(a[l][r-1], mxx[r])

        # b[l][r] = max(mxy[t] for t in [l..r])
        b = [[0]*M for _ in range(M)]
        for l in range(M):
            b[l][l] = mxy[l]
            for r in range(l+1, M):
                b[l][r] = max(b[l][r-1], mxy[r])

        # build the four quadrant-DP arrays
        lu = [[0]*M for _ in range(N)]
        for i in range(N):
            for j in range(M):
                best = cal(i, j)
                if i>0:    best = max(best, lu[i-1][j])
                if j>0:    best = max(best, lu[i][j-1])
                lu[i][j] = best

        ru = [[0]*M for _ in range(N)]
        for i in range(N):
            for j in range(M-1, -1, -1):
                best = cal(i, j+K-1) if j+K-1 < M else 0
                if i>0:    best = max(best, ru[i-1][j])
                if j+1<M:  best = max(best, ru[i][j+1])
                ru[i][j] = best

        ld = [[0]*M for _ in range(N)]
        for i in range(N-1, -1, -1):
            for j in range(M):
                best = cal(i+K-1, j) if i+K-1 < N else 0
                if i+1<N:  best = max(best, ld[i+1][j])
                if j>0:    best = max(best, ld[i][j-1])
                ld[i][j] = best

        rd = [[0]*M for _ in range(N)]
        for i in range(N-1, -1, -1):
            for j in range(M-1, -1, -1):
                best = cal(i+K-1, j+K-1) if i+K-1 < N and j+K-1 < M else 0
                if i+1<N:  best = max(best, rd[i+1][j])
                if j+1<M:  best = max(best, rd[i][j+1])
                rd[i][j] = best

        # now try all 3-square patterns
        ans = 0

        # 1) three horizontal strips
        #    ensure j+K ≤ N-1 ⇒ j < N-K
        for i in range(N):
            for j in range(i+K, N-K):
                total = a[0][i] + a[i+K][j] + a[j+K][N-1]
                if total > ans:
                    ans = total

        # 2) three vertical strips
        for i in range(M):
            for j in range(i+K, M-K):
                total = b[0][i] + b[i+K][j] + b[j+K][M-1]
                if total > ans:
                    ans = total

        # 3) L-shaped splits
        for i in range(N):
            for j in range(M):
                # top split then horizontal split
                if i+K < N and j+1 < M:
                    ans = max(ans, lu[i][j] + ru[i][j+1] + a[i+K][N-1])
                # bottom split then horizontal split
                if i >= K and j+1 < M:
                    ans = max(ans, ld[i][j] + rd[i][j+1] + a[0][i-1])
                # left split then vertical split
                if j+K < M and i+1 < N:
                    ans = max(ans, lu[i][j] + ld[i+1][j] + b[j+K][M-1])
                # right split then vertical split
                if j >= K and i+1 < N:
                    ans = max(ans, ru[i][j] + rd[i+1][j] + b[0][j-1])

        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            K = self.parameter["K"],
            grid = "\n".join(" ".join(map(str, row)) for row in self.parameter["A"]),
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