import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Axis_KCenter_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/K4767
    prompt_template = \
r"""You are given {N} points on a line, labeled from 0 to {N_minus_1}. Their positions (from left to right) are: {X}

Please select a set of {K} distinct points. Try your best to minimize the total distance from all points to their nearest selected point (the distance is the absolute difference between positions).

**Output Format:** Your final answer should be a single line containing the indices of the selected {K} points in any order, separated by spaces."""

    def __init__(self,
                 position_multiple : int = 5,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Axis_KCenter_Environment instance.
        """
        super().__init__(**kwargs)

        self.position_multiple = position_multiple

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

        K = self.parameter["K"] = random.randint(1, N - 1)

        X = self.parameter["X"] = random.sample(range(N * self.position_multiple + 1), N)
        X.sort()


        INF = N * (X[-1] - X[0] + 1)
    
        # Krecompute w[l][r]: cost of one post office for villages l..r (inclusive, 0-indexed)
        w = [[0] * N for _ in range(N)]
        for l in range(N):
            for r in range(l + 1, N):
                m = (l + r) // 2
                w[l][r] = w[l][r - 1] + (X[r] - X[m])
        
        # dp[i][j]: minimum total distance covering the first i villages with j post offices
        dp = [[INF] * (K + 1) for _ in range(N + 1)]
        # d[i][j]: the k giving the optimum for dp[i][j], for Knuth optimization
        d = [[0] * (K + 1) for _ in range(N + 2)]
        
        dp[0][0] = 0
        
        for j in range(1, K + 1):
            d[N + 1][j] = N
            for i in range(N, 0, -1):
                best = INF
                argk = 0
                start = d[i][j - 1]
                end = d[i + 1][j]
                if end > i - 1:
                    end = i - 1
                for k in range(start, end + 1):
                    cost = dp[k][j - 1] + w[k][i - 1]
                    if cost < best:
                        best = cost
                        argk = k
                dp[i][j] = best
                d[i][j] = argk
        
        # Output the result: all N villages with K post offices
        self.parameter["gold_answer"] = dp[N][K]
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            K = self.parameter["K"],
            X = " ".join(map(str, self.parameter["X"])),
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

            selected_points = processed_result

            if len(selected_points) != len(set(selected_points)) :
                return self.rewards["invalid_solution"]
            if len(selected_points) != self.parameter["K"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= u < self.parameter["N"] for u in selected_points) :
                return self.rewards["invalid_solution"]

            answer = sum(min(abs(self.parameter["X"][u] - self.parameter["X"][v]) for v in selected_points) for u in range(self.parameter["N"]))
            gold = self.parameter["gold_answer"]
            assert gold <= answer, "gold should be less than or equal to answer"

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]