import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class BAJBytecomputer_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3558
    prompt_template = \
r"""You are given an array X of length {N}, where each element is initially -1, 0, or +1: {X}
You may perform the following operation any number of times: choose an index i (1 ≤ i < {N}), and update X[i + 1] := X[i + 1] + X[i]. Your goal is to make the array non-decreasing, i.e., X[1] ≤ X[2] ≤ ... ≤ X[{N}]; please output the **minimum number of operations** required to achieve this."""
    
    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = 1.0, incorrect_answer : float = 0.0,
                 **kwargs):
        """
        Initialize the BAJBytecomputer_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        while True :
            distribution = [random.randint(1, N) for _ in range(3)]
            X = self.parameter["X"] = [random.choices([-1, 0, 1], weights = distribution)[0] for _ in range(N)]


            # Compute a suitable "infinity" based on the maximum possible operations:
            # At most 2 operations per element (for N-1 transitions), so 2*N + a small buffer
            INF = 2 * N + 5

            # The three possible values after operations
            val = [-1, 0, 1]
            
            # dp[j] = minimum operations to make the previous element equal to val[j]
            # Initialize for the first element
            prev = [INF] * 3
            prev[X[0] + 1] = 0

            # Iterate through the sequence
            for i in range(1, N):
                curr = [INF] * 3
                x = X[i]
                for j in range(3):
                    ops_so_far = prev[j]
                    if ops_so_far >= INF:
                        continue
                    prev_val = val[j]

                    # 0 operations on x: new_x = x
                    new_x = x
                    if new_x >= prev_val:
                        curr[new_x + 1] = min(curr[new_x + 1], ops_so_far)

                    # 1 operation on x: new_x = x + prev_val
                    new_x = x + prev_val
                    if -1 <= new_x <= 1 and new_x >= prev_val:
                        curr[new_x + 1] = min(curr[new_x + 1], ops_so_far + 1)

                    # 2 operations on x: new_x = x + 2 * prev_val
                    new_x = x + 2 * prev_val
                    if -1 <= new_x <= 1 and new_x >= prev_val:
                        curr[new_x + 1] = min(curr[new_x + 1], ops_so_far + 2)

                prev = curr

            # The answer is the minimum operations to end with any of {-1,0,1}
            ans = min(prev)
            if ans < INF:
                self.parameter["reference_answer"] = ans
                break
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            X = ", ".join("X[{}]={}".format(i + 1, Xi) for i, Xi in enumerate(self.parameter["X"])),
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
                return self.rewards["incorrect_answer"]
        else :
            return self.rewards["wrong_format"]