import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class InversionPairK_Counting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2513
    prompt_template = \
r"""Consider all permutations of the numbers `1` through `{N}`. Your task is to **count how many of them have exactly {K} inversion pairs**.  
Since the number may be large, output the result **modulo {MOD}**.

**Definitions:**
- A **permutation of 1 to {N}** is an arrangement of the numbers `1` through `{N}`, where each number appears exactly once.
- An **inversion pair** in a permutation `a_1, a_2, ..., a_{N}` is a pair of indices `(i, j)` such that `i < j` and `a_i > a_j`.

**Output Format:**
Your final answer should be a single integer — the number of permutations with exactly {K} inversion pairs, **modulo {MOD}**.
Example: `9999` (do **NOT** include the backticks or quotes).
"""
    
    def __init__(self,
                 max_MOD : int = 1000000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the InversionPairK_Counting_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_MOD = max_MOD
        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 1, "N should be greater than or equal to 1"

        K = self.parameter["K"] = random.randint(0, N * (N - 1) // 2)

        MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)


        dpF = [0] * (K + 1)
        dpF[0] = 1
        for i in range(1, N + 1) :
            prefix_sum = [0] * (K + 1)
            prefix_sum[0] = dpF[0]
            for k in range(1, K + 1) :
                prefix_sum[k] = prefix_sum[k - 1] + dpF[k]
            def get_sum(l, r) :
                l = max(l, 0)
                return prefix_sum[r] - (prefix_sum[l - 1] if l > 0 else 0)
            for k in range(min(K, i * (i - 1) // 2) + 1) :
                dpF[k] = get_sum(k - (i - 1), k) % MOD
        self.parameter["reference_answer"] = dpF[K]
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], K = self.parameter["K"], MOD = self.parameter["MOD"])
    

    def _process(self, answer : Optional[str]) -> Optional[int] :
        if answer is not None :
            answer = answer.strip()
            try :
                int_answer = int(answer)
                return int_answer
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if not (0 <= processed_result < self.parameter["MOD"]) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]