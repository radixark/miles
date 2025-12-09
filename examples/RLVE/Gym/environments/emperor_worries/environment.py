import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class EmperorWorries_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4409
    prompt_template = \
r"""There are {N} generals numbered from 0 to {N_minus_1}. The medal requirements are: {A}
Assign medals of various **types** to the generals so that: (1) The medals given to the same general are all of **distinct types** (no duplicate type for one general); (2) Adjacent generals (i and (i+1) mod {N}) share **no common medal type**. What is the **minimum number of medal types** required to satisfy all constraints?"""
    def __init__(self,
                 A_range : int = 2,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the EmperorWorries_Environment instance.
        """
        super().__init__(**kwargs)

        self.A_range = A_range
        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "K" in self.parameter, "K is required in parameter"
        K = self.parameter["K"]
        assert K >= 1, "K should be greater than or equal to 1"

        N = self.parameter["N"] = random.choice((2 * K, 2 * K + 1))
        self.parameter["A"] = [random.randint(1, N * self.A_range) for _ in range(N)]


        A = [None] + self.parameter["A"]  # 1-indexed like the C++ array
        S = 0
        for i in range(1, N + 1):
            S += A[i]

        candidates = []
        for i in range(1, N):
            candidates.append(A[i] + A[i + 1])
        candidates.append(A[1] + A[N])

        K = N // 2
        candidates.append((S + K - 1) // K)  # ceil(S / K) without importing math

        self.parameter["reference_answer"] = max(candidates)
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            A = "; ".join("General {} needs {} medals of distinct types".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
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