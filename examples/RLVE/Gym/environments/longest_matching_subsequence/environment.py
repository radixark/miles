import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Longest_MatchingSubsequence_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1799
    prompt_template = \
r"""You are given an array `A` of length {N}, indexed from 0 to {N_minus_1}. The array is as follows:
{A}

Your task is to select a **strictly increasing sequence of indices** `i_1, i_2, ..., i_k` (0 ≤ i_1 < i_2 < ... < i_k < {N}) such that:
- Let B[1] = A[i_1], B[2] = A[i_2], ..., B[k] = A[i_k] (B's indices are 1-based, while A's indices are 0-based).
- Try your best to **maximize** the number of positions `j` (1 ≤ j ≤ k) such that B[j] = j.

**Output Format:** Your final answer should be a single line containing the selected indices i_1, i_2, ..., i_k, separated by **spaces**. Example: `0 2` (do **NOT** include quotes or backticks); this means you selected indices 0 and 2, with k = 2."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Longest_MatchingSubsequence_Environment instance.
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
        assert N >= 3, "N should be greater than or equal to 3"

        A = self.parameter["A"] = [random.randint(1, N) for _ in range(N - 1)] + [1]
        random.shuffle(A)


        answer = 0
        F = [None] * N
        for i in range(N) :
            if A[i] <= i + 1 :
                F[i] = 1
            for j in range(i) :
                if A[i] - A[j] <= i - j and A[i] > A[j] :
                    if F[j] is not None :
                        val = F[j] + 1
                        if F[i] is None or val > F[i] :
                            F[i] = val
            if F[i] is not None :
                answer = max(answer, F[i])
        assert answer > 0
        self.parameter["gold_answer"] = answer
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            A = "\n".join("A[{}]={}".format(index, value) for index, value in enumerate(self.parameter["A"])),
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
            
            B = [-1]
            for i in range(len(processed_result)) :
                if not (0 <= processed_result[i] < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                if i > 0 and not (processed_result[i - 1] < processed_result[i]) :
                    return self.rewards["invalid_solution"]
                B.append(self.parameter["A"][processed_result[i]])
            answer, gold = sum(int(i == bi) for i, bi in enumerate(B)), self.parameter["gold_answer"]
            assert answer <= gold, "answer should be less than or equal to gold_answer"
            
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * int(answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]