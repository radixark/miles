import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinimumUnconflictedGridKMax_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4251
    prompt_template = \
r"""You are given an {N} × {M} grid of non-negative integers `A[i][j]` (1-indexed). The matrix A is:
{grid}

Choose {N} **distinct** column indices `p[1], p[2], ..., p[{N}]` in the range `[1, {M}]`. For each row `i`, take the value `A[i][p[i]]`; among these {N} values, consider the **{K}-th largest** value; your goal is to **minimize** this {K}-th largest value. Output `p[1] p[2] ... p[{N}]` on a single line, separated by spaces."""
    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinimumUnconflictedGridKMax_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 3, "MAX_N_M should be greater than or equal to 3"

        N = self.parameter["N"] = random.randint(3, MAX_N_M)
        M = self.parameter["M"] = random.randint(N, MAX_N_M)
        self.parameter["K"] = random.randint(1, N)
        self.parameter["A"] = [[random.randint(1, N * M) for j in range(M)] for i in range(N)]


        K = N - self.parameter["K"] + 1  # transform as in the original code

        A = [[0] * (M + 1) for _ in range(N + 1)]
        LIM = -1
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                A[i][j] = self.parameter["A"][i - 1][j - 1]
                if A[i][j] > LIM:
                    LIM = A[i][j]

        def check(x):
            vis = [0] * (M + 1)
            lin = [0] * (M + 1)
            tot = 1
            ans = 0

            def dfs(u, lim):
                for j in range(1, M + 1):
                    if A[u][j] <= lim and vis[j] != tot:
                        vis[j] = tot
                        if lin[j] == 0 or dfs(lin[j], lim):
                            lin[j] = u
                            return True
                return False

            for i in range(1, N + 1):
                if dfs(i, x):
                    ans += 1
                tot += 1
            return ans

        l, r = 1, LIM
        while l < r:
            mid = (l + r) // 2
            if check(mid) >= K:
                r = mid
            else:
                l = mid + 1

        self.parameter["gold_answer"] = l
        assert self.parameter["gold_answer"] > 0, "gold_answer should be positive"
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            K = self.parameter["K"],
            grid = "\n".join(", ".join("A[{}][{}]={}".format(i, j, Aij) for j, Aij in enumerate(row, start = 1)) for i, row in enumerate(self.parameter["A"], start = 1)),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List[int]] :
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

            if len(processed_result) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if len(processed_result) != len(set(processed_result)) :
                return self.rewards["invalid_solution"]
            if not all(1 <= x <= self.parameter["M"] for x in processed_result) :
                return self.rewards["invalid_solution"]
            
            answer, gold = sorted([self.parameter["A"][i][x - 1] for i, x in enumerate(processed_result)], reverse = True)[self.parameter["K"] - 1], self.parameter["gold_answer"]
            assert 0 < gold <= answer, "gold should be less than or equal to answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]