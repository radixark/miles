import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class ValueDiminishingSelection_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2647
    prompt_template = \
r"""You are given {N} items labeled from `0` to `{N_minus_1}`. Each item has a base value W[i] and a diminishing factor R[i]. The list of values and diminishing factors is given as:
{W_and_R}

You must select a sequence of **distinct items** (the order matters). When selecting the i-th item:
- Its effective value is W[i] minus the total of R[j] for all previously selected items j.
- In other words, each item selected **after** i will lose R[i] from their gain due to the diminishing effect.

Your goal is to select a sequence of items to **maximize the total gain**.

**Output Format:** Output a single line containing the indices of the selected items in order, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the ValueDiminishingSelection_Environment instance.
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
        assert N >= 2, "N should be greater than or equal to 2"

        W = self.parameter["W"] = [random.randint(0, N * N // 2) for _ in range(N)]
        R = self.parameter["R"] = [random.randint(1, N) for _ in range(N)]


        P = [(Wi, Ri) for Wi, Ri in zip(W, R)]

        # sort by R descending
        P.sort(key=lambda x: x[1], reverse=True)

        dp = [None] * (N + 1)   # dp[j] = best gain picking j items
        dp[0] = 0
        best = 0                # answer — at least 0 by taking nothing

        for i in range(N):
            W, R = P[i]
            new_dp = dp.copy()          # row i -> row i+1
            for j in range(1, i + 2):   # up to i+1 items can be chosen now
                prev = dp[j - 1]
                if prev is None:
                    continue
                cand = prev + W - R * (j - 1)
                if new_dp[j] is None or cand > new_dp[j]:
                    new_dp[j] = cand
                    if cand > best:
                        best = cand
            dp = new_dp                 # move to next row

        self.parameter["gold_answer"] = best
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            W_and_R = "\n".join("W[{}]={} R[{}]={}".format(i, self.parameter["W"][i], i, self.parameter["R"][i]) for i in range(N)),
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

            if len(processed_result) != len(set(processed_result)) :
                return self.rewards["invalid_solution"]
            if not all(0 <= i < self.parameter["N"] for i in processed_result) :
                return self.rewards["invalid_solution"]
            
            answer, gold = 0, self.parameter["gold_answer"]
            sum_R = 0
            for i in processed_result :
                Wi, Ri = self.parameter["W"][i], self.parameter["R"][i]
                answer += Wi - sum_R
                sum_R += Ri
            answer = max(0, answer)
            assert answer <= gold, "answer should be less than or equal to gold"
            
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                if gold == 0 :
                    assert answer == 0, "If gold is 0, answer should also be 0"
                    return self.rewards["rewarding_weight"] * 1.0
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]