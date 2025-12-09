import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class SalesmanFatigue_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2672
    prompt_template = \
r"""You are given {N} pairs of integers `(S[i], A[i])` for `0 <= i < {N}`, provided as:
{S_and_A}

**Note:** The array `S` is sorted in non-decreasing order: `S[0] <= S[1] <= ... <= S[{N_minus_1}]`

Please select k distinct pairs `i_1, i_2, ..., i_k` and maximize the following expression: `max(S[i_1], S[i_2], ..., S[i_k]) * 2 + A[i_1] + A[i_2] + ... + A[i_k]` (i.e., the sum of the selected A[i] values plus the maximum S[i] value multiplied by 2).
Please compute the **maximum value of this expression** for each k = 1 to {N}.

**Output Format:** Your final answer should be a single line containing {N} integers — the maximum value for each k = 1 to {N} in order, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 5.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the SalesmanFatigueProblem instance.
        """
        super().__init__(**kwargs)

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


        S = self.parameter["S"] = [random.randint(1, max(1, N * N // 2)) for _ in range(N)]
        S.sort()
        A = self.parameter["A"] = [random.randint(1, N) for _ in range(N)]


        v = list(zip(S, A))
        v.sort(key = lambda x : -x[1])

        P = [0] * (N + 1)
        for i in range(N) :
            P[i + 1] = P[i] + v[i][1]

        q = [0] * N
        max_q = 0
        for i in range(N) :
            max_q = max(max_q, 2 * v[i][0])
            q[i] = max_q

        h = [0] * N
        max_h = 0
        for i in range(N - 1, -1, -1) :
            max_h = max(max_h, 2 * v[i][0] + v[i][1])
            h[i] = max_h

        answers = []
        for X in range(1, N + 1) :
            idx = X - 1
            option1 = P[X] + q[idx]
            option2 = P[X - 1] + h[idx]
            answers.append(max(option1, option2))
        
        self.parameter["gold_answer"] = answers
        self.parameter["reference_answer"] = " ".join(map(str, answers))
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            S_and_A = "\n".join("S[{}]={} A[{}]={}".format(i, self.parameter["S"][i], i, self.parameter["A"][i]) for i in range(N)),
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

            if len(processed_result) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["gold_answer"], processed_result)) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]