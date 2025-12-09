import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class LIS_LDS_Concatenation_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1091
    prompt_template = \
r"""You are given an array `A` of length {N}. The values are as follows (indexing starts at 0):
{A}

Your task is to select a strictly increasing sequence of indices `i1, i2, ..., ik` such that:
- `0 ≤ i1 < i2 < ... < ik < {N}`
- Let `a[1], a[2], ..., a[k]` be the values of `A` at the selected indices (i.e., `a[1] = A[i1]`, `a[2] = A[i2]`, ..., `a[k] = A[ik]).` We want the sequence `a[1] < a[2] < ... < a[m] > a[m + 1] > ... > a[k]` for some `m` that satisfies `1 <= m <= k`. In other words, it is allowed for the sequence to first be strictly increasing, then strictly decreasing. It is also allowed for the sequence to be entirely strictly increasing or entirely strictly decreasing.
- Your goal is to **maximize the length** of the selected sequence `k`.

**Output Format:**
Your final answer should be a single line containing the selected indices `i1, i2, ..., ik`, separated by **spaces**.
Example: `0 2 3` (do **NOT** include the backticks or quotes); this means the sequence has length `k = 3`, with `i1 = 0`, `i2 = 2`, and `i3 = 3`.
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the WeightedLIS_Environment instance.
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
        assert N >= 1, "N should be greater than or equal to 1"

        assert "MAX" in self.parameter, "MAX is required in parameter"
        MAX = self.parameter["MAX"]
        assert MAX >= 1, "MAX should be greater than or equal to 1"

        array = self.parameter["array"] = [random.randint(0, MAX) for _ in range(N)]
        assert len(self.parameter["array"]) == self.parameter["N"], "array should have the same length as N"
        
        
        F, G = [0] * N, [0] * N
        for i in range(N) :
            F[i] = 1
            for j in range(i) :
                if array[j] < array[i] :
                    F[i] = max(F[i], F[j] + 1)
        for i in range(N - 1, -1, -1) :
            G[i] = 1
            for j in range(i + 1, N) :
                if array[i] > array[j] :
                    G[i] = max(G[i], G[j] + 1)

        Answer = 0
        for i in range(N) :
            Answer = max(Answer, F[i] + G[i] - 1)
        self.parameter["gold_answer"] = Answer
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], A = " ".join(map(str, self.parameter["array"])))


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

            values = []
            for i in range(len(processed_result)) :
                if not (0 <= processed_result[i] < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                if i > 0 and not (processed_result[i - 1] < processed_result[i]) :
                    return self.rewards["invalid_solution"]
                values.append(self.parameter["array"][processed_result[i]])
            
            increasing, decreasing = [False] * self.parameter["N"], [False] * self.parameter["N"]
            for i in range(len(values)) :
                if i :
                    increasing[i] = increasing[i - 1] and (values[i - 1] < values[i])
                else :
                    increasing[i] = True
            found = False
            for i in range(len(values) - 1, -1, -1) :
                if i < len(values) - 1 :
                    decreasing[i] = decreasing[i + 1] and (values[i] > values[i + 1])
                else :
                    decreasing[i] = True
                if increasing[i] and decreasing[i] :
                    found = True
                    break
            
            if not found :
                return self.rewards["invalid_solution"]
            
            assert len(processed_result) <= self.parameter["gold_answer"], "The length of the answer should be less than or equal to the gold answer"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((len(processed_result) / self.parameter["gold_answer"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * int(len(processed_result) == self.parameter["gold_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]