import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class KloBlocks_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3503
    prompt_template = \
r"""You have an array A of {N} integers, initially it is: {A}
You can perform any number of actions. One action is to pick one item that is **greater than** {K}, subtract 1 from it, and add 1 to an **adjacent** item (either to the left or right, if such an item exists). 
Please maximize the length of the longest contiguous subarray where each item is **greater than or equal to** {K}; output its length."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the KloBlocks_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "correct_answer": correct_answer,
            "wrong_answer": wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        while True :
            A = self.parameter["A"] = [random.randint(1, 2 * N) for _ in range(N)]
            min_A, max_A = min(A), max(A)
            if not (min_A + 1 <= max_A - 1) :
                continue
            K = self.parameter["K"] = random.randint(min_A + 1, max_A - 1)


            # b[0] = 0, b[i] = prefix sum of (A[j] - K) up to j = i
            b = [0] * (N + 1)
            stack = []  # will store indices with strictly decreasing b-values
            ans = 0
            
            # Forward pass: build b[], track any prefix >= 0 and build monotonic stack
            for i in range(1, N + 1):
                b[i] = b[i-1] + A[i-1] - K
                if b[i] >= 0:
                    # we can take the whole prefix 1..i
                    ans = i
                # maintain stack of indices where b is strictly decreasing
                if not stack or b[i] < b[stack[-1]]:
                    stack.append(i)
            
            # Backward pass: match later indices i with earlier minima in stack
            for i in range(N, 0, -1):
                # while we can form a non-negative sum from stack[-1]+1 .. i
                while stack and b[i] - b[stack[-1]] >= 0:
                    ans = max(ans, i - stack[-1])
                    stack.pop()

            if ans != 1 and ans != N :
                self.parameter["reference_answer"] = ans
                break
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], A = " ".join(map(str, self.parameter["A"])), K = self.parameter["K"])


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