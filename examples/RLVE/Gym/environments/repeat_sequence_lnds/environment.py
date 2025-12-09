import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class RepeatSequenceLNDS_Environment(VerifiableEnvironment):
    prompt_template = \
r"""You are given an array that repeats every {n} elements. The initial pattern is: {a}. This pattern repeats {T} times, creating a total array length of {nT}.

For example, if the initial pattern is [1, 3, 2] and it repeats 2 times, the full array would be [1, 3, 2, 1, 3, 2].

Find the length of the longest non-decreasing subsequence (not necessarily contiguous) in this repeated array.

Your answer should be a single integer."""

    def __init__(self,
                 wrong_format: float = -1.0, incorrect_solution: float = 0.0, correct_solution: float = 1.0,
                 **kwargs):
        """
        Initialize the RepeatSequenceLNDS_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "incorrect_solution": incorrect_solution,
            "correct_solution": correct_solution,
        }

    def _generate(self) -> None:
        assert "n" in self.parameter, "n is required in parameter"
        n = self.parameter["n"]
        assert n >= 2, "n must be at least 2"

        assert "MAX_T" in self.parameter, "MAX_T is required in parameter"
        MAX_T = self.parameter["MAX_T"]
        assert MAX_T >= 2, "MAX_T must be at least 2"
        
        T = self.parameter["T"] = random.randint(2, MAX_T)

        # Generate the initial array of length n
        self.parameter["a"] = a = [random.randint(1, n) for _ in range(n)]
        
        # Calculate the reference answer using the provided algorithm
        self.parameter["reference_answer"] = self._calculate_longest_nds(a, n, T)

    def _calculate_longest_nds(self, a, n, T):
        """
        Calculate the longest non-decreasing subsequence using the provided algorithm.
        Source: https://codeforces.com/contest/582/submission/282761264
        """
        # Initialize frequency array for elements (1 to max(a))
        s = [0] * (max(a) + 1)
        d = [0] * (max(a) + 1)

        # Count the frequency of each element in the initial array
        for i in a:
            d[i] += 1

        # Calculate the longest non-decreasing subsequence
        # Iterate over the array repeated min(T, 2 * n) times
        for i in a * min(T, 2 * n):
            # Update the dynamic programming array
            s[i] = max(s[:i + 1]) + 1

        # Calculate the maximum length of the subsequence
        # Consider extending the subsequence with full repetitions of the most frequent element
        return max(s) + max((T - n * 2) * max(d), 0)

    def _prompt_generate(self) -> str:
        n, T = self.parameter["n"], self.parameter["T"]
        return self.prompt_template.format(
            n=n,
            T=T,
            nT=n * T,
            a=str(self.parameter["a"])
        )

    def _process(self, answer: Optional[str]) -> Optional[int]:
        if answer is not None:
            answer = answer.strip()
            try:
                int_answer = int(answer)
                return int_answer
            except ValueError:
                return None
        else:
            return None

    def scorer(self, output: str) -> float:
        processed_result = self.processor(output)
        if processed_result is not None:
            if processed_result == self.parameter["reference_answer"]:
                return self.rewards["correct_solution"]
            else:
                return self.rewards["incorrect_solution"]
        else:
            return self.rewards["wrong_format"] 