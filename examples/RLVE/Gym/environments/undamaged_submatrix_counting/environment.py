import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class UndamagedSubmatrixCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3400
    prompt_template = \
r"""You are given a matrix of size {N} × {M}, where each element is either `0` or `1`. Please count the number of **contiguous non-empty submatrices** that consist entirely of `1`s. The matrix is:
{matrix}

Note:
- Two submatrices are considered different if they differ in position, even if they contain the identical elements.
- The whole matrix itself is also considered a submatrix.
- **Output Format:** A single non-negative integer — the total number of all-one submatrices."""


    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the UndamagedSubmatrixCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    
    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)
        one_probability = random.random()
        A = self.parameter["matrix"] = [[1 if random.random() < one_probability else 0 for _ in range(M)] for _ in range(N)]


        # f[j] stores the most recent row index where column j had a 0 (initialized to -1)
        f = [-1] * M
        ans = 0

        # Process each row
        for i in range(N):
            # Monotonic stack: stores pairs (column_index, height)
            stack = []
            # sum_arr[k] stores the cumulative count for stack up to index k
            sum_arr = []

            for j in range(M):
                # Update last-zero position for this column
                if A[i][j] == 0:
                    f[j] = i
                # Height of consecutive ones ending at (i, j)
                height = i - f[j]

                # Pop columns with greater height to maintain non-decreasing heights
                while stack and stack[-1][1] > height:
                    stack.pop()
                    sum_arr.pop()

                # Compute contribution for this column
                if not stack:
                    # All columns to the left are shorter; width = j+1
                    total = height * (j + 1)
                else:
                    # Extend from the last column in the stack
                    prev_total = sum_arr[-1]
                    prev_idx, _ = stack[-1]
                    total = prev_total + height * (j - prev_idx)

                # Push current column onto the stack
                stack.append((j, height))
                sum_arr.append(total)

                # Accumulate into answer
                ans += total

        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            matrix = "\n".join("".join(map(str, row)) for row in self.parameter["matrix"]),
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
            if processed_result < 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                if processed_result == 0 :
                    return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == 0)
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]