import random
from collections import deque
from typing import Optional, Tuple
from Gym.environment import VerifiableEnvironment


class LongestMaxDiffBoundedInterval_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3512
    prompt_template = \
r"""You are given an array A of length {N}: {A}

Please find the longest **contiguous** subarray A[l : r] (from index `l` to `r - 1`, inclusive) such that the **maximum difference between any two elements** in the subarray is at most {K}. Output `l` and `r`, separated by a space."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the LongestMaxDiffBoundedInterval_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "unsuccessful_solution" : unsuccessful_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        A = self.parameter["A"] = [random.randint(0, N) for _ in range(N)]
        K = self.parameter["K"] = random.randint(0, max(max(A) - min(A) - 1, 0))


        # Deques to maintain indices of potential max/min in the current window
        max_deque = deque()  # will store indices of A in decreasing order of values
        min_deque = deque()  # will store indices of A in increasing order of values

        left = 0
        answer = 0

        for right, value in enumerate(A):
            # Maintain max_deque: pop smaller elements from the tail
            while max_deque and A[max_deque[-1]] <= value:
                max_deque.pop()
            max_deque.append(right)

            # Maintain min_deque: pop larger elements from the tail
            while min_deque and A[min_deque[-1]] >= value:
                min_deque.pop()
            min_deque.append(right)

            # Shrink window from the left until the max − min ≤ K
            while A[max_deque[0]] - A[min_deque[0]] > K:
                # Advance left past whichever extreme comes first
                if max_deque[0] < min_deque[0]:
                    left = max_deque[0] + 1
                    max_deque.popleft()
                else:
                    left = min_deque[0] + 1
                    min_deque.popleft()

            # Update the answer with the current valid window size
            answer = max(answer, right - left + 1)

        assert answer > 0, "The answer should be greater than 0"
        self.parameter["gold_answer"] = answer
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
            K = self.parameter["K"],
        )


    def _process(self, answer : Optional[str]) -> Optional[Tuple[int, int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                l, r = map(int, answer.split())
                return l, r
            except :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            l, r = processed_result
            if not (0 <= l < r <= self.parameter["N"]) :
                return self.rewards["invalid_solution"]
            if max(self.parameter["A"][l : r]) - min(self.parameter["A"][l : r]) > self.parameter["K"] :
                return self.rewards["unsuccessful_solution"]

            answer, gold = r - l, self.parameter["gold_answer"]
            assert 0 < answer <= gold, "The answer should not be greater than the gold answer"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * int(answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]