import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class DistinctArrayPermutation_Environment(VerifiableEnvironment):
    prompt_template = \
r"""You are given an array A with {N} distinct integers (1-indexing): {array}

Construct an array B by permuting A such that for every non-empty proper subset of indices S = {{x1, x2, ..., xk}} (1 ≤ xi ≤ {N}, 0 < k < {N}) the sums of elements on that positions in A and B are different.

Your final answer should be a single line containing the permuted array B's elements in order, separated by spaces."""

    def __init__(self,
                 wrong_format: float = -1.0,
                 invalid_solution: float = -0.5,
                 incorrect_solution: float = 0, 
                 correct_solution: float = 1.0,
                 **kwargs):
        """
        Initialize the DistinctArrayPermutation_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "incorrect_solution": incorrect_solution,
            "correct_solution": correct_solution,
        }

    def _find_valid_permutation(self, arr: List[int]) -> List[int]:
        """
        Find a valid permutation of arr such that all subset sums are different.
        Uses the elegant solution: sort indices by values, then cyclically assign next value.
        """
        n = len(arr)
        
        # Sort indices by the values in the array
        p = sorted([i for i in range(n)], key=lambda x: arr[x])
        
        # Create the permutation
        b = [0] * n
        for i in range(n):
            b[p[i]] = arr[p[(i + 1) % n]]
        
        return b

    def _is_valid_permutation(self, arr_a: List[int], arr_b: List[int]) -> bool:
        """
        Check if arr_b is a valid permutation that satisfies the condition.
        """
        n = len(arr_a)
        
        # Check if it's actually a permutation
        if sorted(arr_a) != sorted(arr_b):
            return False
        
        # Check all non-empty proper subsets
        for mask in range(1, (1 << n) - 1):  # From 1 to 2^n - 2
            sum_a = 0
            sum_b = 0
            for i in range(n):
                if mask & (1 << i):
                    sum_a += arr_a[i]
                    sum_b += arr_b[i]
            
            if sum_a == sum_b:
                return False
        
        return True


    def _generate(self) -> None:
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be at least 3"

        # Generate array with distinct integers using max_value = 2*N
        # Yes, random.sample() returns a list of N unique elements sampled from range(max_value)
        self.parameter["array"] = random.sample(range(2 * N), N)
        self.parameter["reference_answer"] = " ".join(map(str, self._find_valid_permutation(self.parameter["array"])))


    def _prompt_generate(self) -> str:
        return self.prompt_template.format(
            N = self.parameter["N"],
            array = " ".join(map(str, self.parameter["array"])),
        )


    def _process(self, answer: Optional[str]) -> Optional[List[int]]:
        if answer is not None:
            answer = answer.strip()
            try:
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError:
                return None  # Invalid answer format
        else:
            return None  # Invalid answer format


    def scorer(self, output: str) -> float:
        processed_result = self.processor(output)
        if processed_result is not None:
            assert isinstance(processed_result, list), "processed_result should be a list"
            # Check if it's a valid permutation
            if sorted(processed_result) != sorted(self.parameter["array"]):
                return self.rewards["invalid_solution"]

            # Check if it satisfies the distinct subset sum condition
            if self._is_valid_permutation(self.parameter["array"], processed_result):
                return self.rewards["correct_solution"]  # Correct solution
            else:
                return self.rewards["incorrect_solution"]  # Invalid permutation

        else:
            return self.rewards["wrong_format"] 