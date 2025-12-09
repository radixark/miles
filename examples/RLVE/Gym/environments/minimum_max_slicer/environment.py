import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Minimum_MaxSlicer_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1182
    prompt_template = \
r"""You are given an array A of length {N}. The values are as follows (indexing starts at 0):
{A}

You may divide these items (in order) into {M} **consecutive batches**. Let end[1], end[2], ..., end[{M}] (0 <= end[1] < end[2] < ... < end[{M}] = {N} - 1 = {N_minus_1}) represent the last index of each batch. This means:
- Batch 1 contains items from index 0 to end[1]
- Batch 2 contains items from index end[1] + 1 to end[2]
- ...
- Batch {M} contains items from index end[{M_minus_1}] + 1 to end[{M}] (which is {N_minus_1})

Try your best to **minimize the maximum sum** among all batches. In other words, minimize: max(S[1], S[2], ..., S[{M}]), where each S[i] is the sum of A values in batch i.

**Output Format:**
Your final answer should be a single line containing end[1], end[2], ..., end[{M}] (with end[{M}] always equal to {N_minus_1}), separated by **spaces**.
Example: `{first_M_minus_1_indices} {N_minus_1}` (do **NOT** include the backticks or quotes); this means: end[1] = 0, ..., end[{M_minus_1}] = {M_minus_2}, and end[{M}] = {N_minus_1}. So, the first {M_minus_1} batches each contain one item, and the last batch contains the remaining items.
"""

    def __init__(self,
                 M_range_coefficient : int = 2,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = +3.0,
                 **kwargs) :
        """
        Initialize the Minimum_MaxSlicer_Environment instance.
        """
        super().__init__(**kwargs)
        self.M_range_coefficient = M_range_coefficient

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
        assert N >= 4, "N must be at least 4"

        M = self.parameter["M"] = random.randint(3, max(3, N // self.M_range_coefficient))
        assert M < N, "M must be less than N"

        A = self.parameter["A"] = [random.randint(1, N) for i in range(N)]


        left, right = min(A), sum(A)
        while left < right :
            mid = (left + right) // 2
            def check(d) :
                now_sum, index, counting = 0, 0, 1
                while True :
                    if now_sum + A[index] <= d :
                        now_sum += A[index]
                    else :
                        counting += 1
                        if A[index] <= d :
                            now_sum = A[index]
                        else :
                            return False
                    index += 1
                    if index == N :
                        break
                return counting <= M
            if check(mid) :
                right = mid
            else :
                left = mid + 1
        self.parameter["gold_answer"] = left
        assert self.parameter["gold_answer"] > 0, "gold_answer must be greater than 0"

        ends = []
        def get_ends(d) :
            now_sum, index = 0, 0
            while True :
                if now_sum + A[index] <= d :
                    now_sum += A[index]
                else :
                    ends.append(index - 1)
                    now_sum = A[index]
                index += 1
                if index == N :
                    ends.append(index - 1)
                    break
        get_ends(left)
        if len(ends) < M :
            missing = sorted(set(range(N)) - set(ends))
            ends += missing[: M - len(ends)]
            ends.sort()
        assert len(ends) == M
        self.parameter["reference_answer"] = " ".join(map(str, ends))
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        M = self.parameter["M"]
        return self.prompt_template.format(
            N = N,
            M = M,
            N_minus_1 = N - 1,
            M_minus_1 = M - 1,
            M_minus_2 = M - 2,
            A = "\n".join("A[{}]={}".format(i, self.parameter["A"][i]) for i in range(N)),
            first_M_minus_1_indices = " ".join(map(str, range(M - 1))),
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

            N = self.parameter["N"]

            ends = processed_result
            if len(ends) != self.parameter["M"] :
                return self.rewards["invalid_solution"]
            for i in range(len(ends)) :
                if not (0 <= ends[i] < N) :
                    return self.rewards["invalid_solution"]
                if i and not (ends[i - 1] < ends[i]) :
                    return self.rewards["invalid_solution"]
            if ends[-1] != N - 1 :
                return self.rewards["invalid_solution"]
            
            answer = sum(self.parameter["A"][index] for index in range(ends[0] + 1))
            for i in range(1, len(ends)) :
                answer = max(answer, sum(self.parameter["A"][index] for index in range(ends[i - 1] + 1, ends[i] + 1)))
            gold = self.parameter["gold_answer"]
            assert gold <= answer, "answer should be greater than or equal to gold"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]