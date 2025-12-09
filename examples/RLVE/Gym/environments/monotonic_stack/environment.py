import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MonotonicStack_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2866
    prompt_template = \
r"""You are given an array A indexed from `1` to `{N}`: {A}

For each 1 ≤ i ≤ {N}, define C[i] as the number of indices j such that:
- i + 1 ≤ j ≤ {N}, and
- For every index k such that i + 1 ≤ k ≤ j, we have A[i] > A[k].

Tell me the value of C[1] + C[2] + ... + C[{N}]."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MonotonicStack_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> str :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N must be at least 3"

        self.parameter["A"] = A = [random.randint(1, N) for _ in range(N)]

        S = []  # monotonic decreasing stack of heights
        ans = 0

        for t in A:
            while S and S[-1] <= t:
                S.pop()
            ans += len(S)
            S.append(t)

        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], A = ", ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"], start = 1)))


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