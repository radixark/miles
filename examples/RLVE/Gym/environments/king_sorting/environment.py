import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class KingSorting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1080
    prompt_template = \
r"""You are given `{N} + 1 = {N_plus_1}` pairs of integers: `(A[0], B[0])`, `(a[1], b[1])`, `(a[2], b[2])`, ..., `(a[{N}], b[{N}])`
{values}

Your task is to **rearrange the {N} pairs** `(a[i], b[i])` for `1 ≤ i ≤ {N}` in some order (there are `{N}!` possible permutations). After rearrangement, define the new sequence of `{N_plus_1}` pairs as: `(A[0], B[0])`, `(A[1], B[1])`, ..., `(A[{N}], B[{N}])`, where `(A[i], B[i])` comes from the chosen permutation for `i ≥ 1`.

Your goal is to **minimize** the following value: `max ( A[0] * A[1] * ... * A[i - 1] // B[i] | 1 ≤ i ≤ {N} )` (Note: `//` means **integer division**, i.e., rounded down just like in Python).
That is, for each `i` from `1` to `{N}`, compute the product of all previous `A` values (`A[0]` to `A[i - 1]`) divided by `B[i]`, take the maximum of these, and find a permutation that minimizes this maximum.

Output Format:
Your final answer should be a single line containing a permutation of integers from `1` to `{N}` (space-separated).
Example: `{REVERSE_INDICES}` (do **NOT** include the backticks or quotes); this means:  `(A[1], B[1]) = (a[{N}], b[{N}])`, `(A[2], B[2]) = (a[{N_minus_1}], b[{N_minus_1}])`, ..., `(A[{N}], B[{N}]) = (a[1], b[1])`
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = +5.0,
                 **kwargs) :
        """
        Initialize the KingSorting_Environment instance.
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

        assert "MAX_A_B" in self.parameter, "MAX_A_B is required in parameter"
        MAX_A_B = self.parameter["MAX_A_B"]
        assert MAX_A_B >= 1, "MAX_A_B should be greater than or equal to 1"

        self.parameter["array"] = [{"index" : index, "A" : random.randint(1, MAX_A_B), "B" : random.randint(1, MAX_A_B)} for index in range(0, N + 1)]

        array = self.parameter["array"].copy()
        array[1 :] = sorted(array[1 :], key = lambda x : x["A"] * x["B"])
        Ans = 0
        Mult = array[0]["A"]
        for i in range(1, N + 1) :
            Ans = max(Ans, Mult // array[i]["B"])
            Mult *= array[i]["A"]
        self.parameter["gold_answer"] = Ans
        self.parameter["reference_answer"] = " ".join([str(item["index"]) for item in array[1 :]])
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        array = self.parameter["array"]
        return self.prompt_template.format(
            N = N,
            N_plus_1 = N + 1,
            N_minus_1 = N - 1,
            values = "\n".join(["(A[0], B[0]) = ({}, {})".format(array[0]["A"], array[0]["B"])] + ["(a[{}], b[{}]) = ({}, {})".format(i, i, array[i]["A"], array[i]["B"]) for i in range(1, N + 1)]),
            REVERSE_INDICES = " ".join([str(i) for i in range(N, 1 - 1, -1)]),
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
            if len(set(processed_result)) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            for i in processed_result :
                if not (1 <= i <= self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
            
            array = self.parameter["array"].copy()
            array[1 :] = [array[i] for i in processed_result]
            answer = 0
            Mult = array[0]["A"]
            for i in range(1, self.parameter["N"] + 1) :
                assert array[i]["index"] == processed_result[i - 1]
                answer = max(answer, Mult // array[i]["B"])
                Mult *= array[i]["A"]
            
            assert self.parameter["gold_answer"] <= answer, "answer should be greater than or equal to gold"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if answer == 0 :
                    assert self.parameter["gold_answer"] == 0, "gold_answer should be 0 if answer is 0"
                    return self.rewards["rewarding_weight"]
                return self.rewards["rewarding_weight"] * ((self.parameter["gold_answer"] / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == self.parameter["gold_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]