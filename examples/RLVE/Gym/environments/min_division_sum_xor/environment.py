import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinDivisionSumXor_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3646
    prompt_template = \
r"""You are given {N} numbers A[1], A[2], ..., A[{N}]. The values are given as:
{A}

You may divide these numbers (in order) into some **consecutive batches**. Let the total number of batches be k (we must have 1 ≤ k ≤ {K}), and let end[1], end[2], ..., end[k] (1 ≤ end[1] < end[2] < ... < end[k] = {N}) denote the last index in each batch. This means:
- Batch 1 contains A[1] to A[end[1]]
- Batch 2 contains A[end[1] + 1] to A[end[2]]
- ...
- Batch k contains A[end[k−1] + 1] to A[end[k]] (with end[k] = {N})

Define the cost of one such division as follows:
- First compute the sum of values in each batch.
- Then take the **bitwise OR** of all batch sums. That is the cost.

Please find a batch division (with 1 ≤ k ≤ {K}) that **minimizes the total cost**.

**Output Format:**
A single line containing `end[1] end[2] ... end[k]`, separated by spaces (with `end[k]` always equal to {N}).
Example: `1 2 {N}` — this means:
- There are 3 batches,
- First batch ends at index 1,
- Second ends at index 2,
- Third ends at index {N} and includes the remaining numbers."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_beta : float = 5.0, rewarding_weight : float = 1.0,
                 **kwargs) :
        """
        Initialize the MinDivisionSumXor_Environment instance.
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
        assert N >= 2, "N should be greater than or equal to 2"

        A = self.parameter["A"] = [random.randint(0, N * N) for _ in range(N)]
        K = self.parameter["K"] = random.randint(2, N)


        # Prefix sums for quick segment sum
        prefix = [0] * (N + 1)
        for i in range(1, N + 1):
            prefix[i] = prefix[i - 1] + A[i - 1]

        def check(idx, ans):
            # DP f[i]: min groups to cover first i sculptures
            INF = N + 1
            f = [INF] * (N + 1)
            f[0] = 0
            mask = ans
            for i in range(1, N + 1):
                # try last segment [j, i)
                for j in range(i - 1, -1, -1):
                    seg_sum = prefix[i] - prefix[j]
                    if ((seg_sum >> idx) & 1) != 0:
                        continue
                    if (((seg_sum >> idx) << idx) | mask) != mask:
                        continue
                    if f[j] + 1 < f[i]:
                        f[i] = f[j] + 1
            return f[N] <= K

        ans = 0
        for idx in range(sum(A).bit_length() + 1, -1, -1):
            ok = check(idx, ans)
            # if not possible to keep this bit zero, set it
            if not ok:
                ans |= (1 << idx)
        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = "\n".join("A[{}]={}".format(i + 1, Ai) for i, Ai in enumerate(self.parameter["A"])),
            K = self.parameter["K"],
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                if not answer_array :
                    return None
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
            if not (1 <= len(ends) <= self.parameter["K"]) :
                return self.rewards["invalid_solution"]
            for i in range(len(ends)) :
                if not (1 <= ends[i] <= N) :
                    return self.rewards["invalid_solution"]
                if i and not (ends[i - 1] < ends[i]) :
                    return self.rewards["invalid_solution"]
            if ends[-1] != N :
                return self.rewards["invalid_solution"]
            A = [None] + self.parameter["A"]
            
            answer = 0
            last = 0
            for end in ends :
                batch_sum = sum(A[last + 1 : end + 1])
                answer |= batch_sum
                last = end
            gold = self.parameter["gold_answer"]
            assert gold <= answer, "Gold answer should be less than or equal to the computed answer"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if answer == 0 :
                    assert gold == 0, "If answer is 0, gold should also be 0"
                    return self.rewards["rewarding_weight"] * 1.0
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]