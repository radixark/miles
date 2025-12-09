import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SubarrayXorSum_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3760
    prompt_template = \
r"""You are given an array A of {N} integers: {A}
This array has {N} × ({N} + 1) / 2 contiguous subarrays. For each subarray, compute the bitwise XOR of its elements, then output the **sum** of all these subarray XOR values."""


    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the SubarrayXorSum_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        A = self.parameter["A"] = [random.randint(0, N) for _ in range(N)]


        # Use only as many bits as needed
        or_all = 0
        for x in A:
            or_all |= x
        B = or_all.bit_length()

        def compute() -> int :
            # If all zeros, the answer is zero
            if B == 0:
                return 0

            cnt_zero = [1] * B   # counts of previous prefixes with bit j == 0 (include s[0]=0)
            cnt_one = [0] * B    # counts of previous prefixes with bit j == 1
            prefix = 0
            ans = 0

            for x in A:
                prefix ^= x
                for j in range(B - 1, -1, -1):
                    bit = (prefix >> j) & 1
                    if bit:
                        ans += (1 << j) * cnt_zero[j]
                        cnt_one[j] += 1
                    else:
                        ans += (1 << j) * cnt_one[j]
                        cnt_zero[j] += 1

            return ans
        self.parameter["reference_answer"] = compute()
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = ", ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"], start = 1)),
        )


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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]