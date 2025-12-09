import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MaxMultSplit_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1018
    prompt_template = \
r"""You are given a string of digits `S` of length {N}:
{string}

Your task is to divide this string into exactly {K_plus_1} non-empty, non-overlapping parts (from left to right, maintaining original order), such that the **product** of the resulting integer values is **maximized**.

Specifically, split the string into substrings: s_1, ..., s_{K_plus_1}, where:
- Each part s_i is a contiguous non-empty substring of `S`,
- The concatenation s_1 + ... + s_{K_plus_1} = S (here + means string concatenation),
- The value `int(s_1) * ... * int(s_{K_plus_1})` is as large as possible.

Output Format:
Your final answer should be a single line containing the {K_plus_1} parts, separated by **spaces**.
Example: `31 2` (do **NOT** include the backticks or quotes); this means the string "312" is split into two parts: "31" and "2".
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the MaxMultSplit_Environment instance.
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

        assert "K" in self.parameter, "K is required in parameter"
        K = self.parameter["K"]
        assert K >= 1, "K should be greater than or equal to 1"

        assert K + 1 <= N, "K + 1 should be less than or equal to N"

        string = self.parameter["string"] = "".join([str(random.randint(1, 9)) for _ in range(N)])
        

        # Dynamic programming to find the maximum product of split
        # dpF[k][i] = max{int(string[i : j]) * dpF[k - 1][j] | j in [i + 1, N - 1]}
        dpF = [[0] * N for _ in range(K + 1)]
        for k in range(0, K + 1) :
            for i in range(N) :
                if not k :
                    dpF[0][i] = int(string[: i + 1])
                else :
                    for j in range(1, i + 1) :
                        dpF[k][i] = max(dpF[k][i], int(string[j : i + 1]) * dpF[k - 1][j - 1])
        self.parameter["gold_answer"] = dpF[K][N - 1]
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            K_plus_1 = self.parameter["K"] + 1,
            string = self.parameter["string"],
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
    

    def scorer(self, output : str) -> str :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            if len(processed_result) != self.parameter["K"] + 1 :
                return self.rewards["invalid_solution"]
            if "".join([str(a) for a in processed_result]) != self.parameter["string"] :
                return self.rewards["invalid_solution"]

            answer = 1
            for val in processed_result :
                assert isinstance(val, int), "val should be an integer"
                answer *= val
            assert answer <= self.parameter["gold_answer"], "answer should be less than or equal to gold_answer"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / self.parameter["gold_answer"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == self.parameter["gold_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]