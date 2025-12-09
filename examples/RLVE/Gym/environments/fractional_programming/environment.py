import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class FractionalProgramming_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P10505
    prompt_template = \
r"""You are given two arrays `A` and `B`, each containing {N} integers:
{A_and_B}

Please select {K} **distinct indices** `i_1, ..., i_{K}` to maximize the value of `(A[i_1] + ... + A[i_{K}]) / (B[i_1] + ... + B[i_{K}])`

**Output Format:** Your final answer should be a single line containing the {K} selected indices in any order, separated by spaces."""


    def __init__(self,
                 max_proportion : int = 2,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the FractionalProgramming_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_proportion = max_proportion

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

        K = self.parameter["K"] = random.randint(2, N - 1)

        B = self.parameter["B"] = [random.randint(1, N) for _ in range(N)]
        A = self.parameter["A"] = [random.randint(1, self.max_proportion * b) for b in B]

        l, r = 0.0, max(a / b for a, b in zip(A, B) if b)
        solution = None
        for _ in range(256) :
            mid = (l + r) / 2
            indices = list(range(N))
            indices.sort(key = lambda index : A[index] - mid * B[index], reverse = True)
            if sum(A[index] - mid * B[index] for index in indices[: K]) >= 0 :
                l = mid
                solution = indices[: K].copy()
            else :
                r = mid
        self.parameter["reference_answer"] = " ".join(map(str, solution))

        self.parameter["gold_SumA"], self.parameter["gold_SumB"] = sum(A[index] for index in solution), sum(B[index] for index in solution)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            K = self.parameter["K"],
            A_and_B = "\n".join("A[{}]={} B[{}]={}".format(i, self.parameter["A"][i], i, self.parameter["B"][i]) for i in range(self.parameter["N"])),
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

            selected_indices = processed_result

            if len(selected_indices) != self.parameter["K"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= index < self.parameter["N"] for index in selected_indices) :
                return self.rewards["invalid_solution"]
            if len(selected_indices) != len(set(selected_indices)) :
                return self.rewards["invalid_solution"]

            answer_SumA, answer_SumB = sum(self.parameter["A"][index] for index in selected_indices), sum(self.parameter["B"][index] for index in selected_indices)
            gold_SumA, gold_SumB = self.parameter["gold_SumA"], self.parameter["gold_SumB"]
            # gold_SumA / gold_SumB >= answer_SumA / answer_SumB   <=>   gold_SumA * answer_SumB >= answer_SumA * gold_SumB
            assert gold_SumA * answer_SumB >= answer_SumA * gold_SumB, "gold_SumA * answer_SumB should be greater than or equal to answer_SumA * gold_SumB"

            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                # (answer_SumA / answer_SumB) / (gold_SumA / gold_SumB) = (answer_SumA * gold_SumB) / (answer_SumB * gold_SumA)
                return self.rewards["rewarding_weight"] * (((answer_SumA * gold_SumB) / (answer_SumB * gold_SumA)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * ((answer_SumA * gold_SumB) == (answer_SumB * gold_SumA))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]