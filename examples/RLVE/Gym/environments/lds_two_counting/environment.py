import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class LDSTwo_Counting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Consider a permutation A[1], A[2], ..., A[{N}] of the integers 1 through {N} that satisfies the following conditions:
- `A` is a **permutation**, meaning each integer from 1 to {N} appears **exactly once**.
- The value at position {X} is fixed: A[{X}] = {Y}.
- The permutation must **not contain any decreasing subsequence of length 3**. That is, there must not exist indices 1 <= a < b < c <= {N} such that A[a] > A[b] > A[c].

Please count the number of such permutations.

**Output Format:** Your final answer should be a single integer — the total number of valid permutations."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the LDSTwo_Counting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        X = self.parameter["X"] = random.randint(1, N)
        Y = self.parameter["Y"] = random.randint(1, N)


        def C(n : int, m : int) :
            if n < m or m < 0 :
                return 0
            result = 1
            for i in range(m) :
                result = result * (n - i) // (i + 1)
            return result

        def go(sx, sy, tx, ty) :
            return C(tx - sx + ty - sy, tx - sx)

        def solve(sx, sy, tx, ty) :
            return go(sx, sy, tx, ty) - go(sx, sy, ty + 1, tx - 1)

        if Y < X :
            X, Y = Y, X
        self.parameter["reference_answer"] = solve(0, 0, X - 1, Y - 1) * solve(X, Y, N, N)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            X = self.parameter["X"],
            Y = self.parameter["Y"],
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

            if self.parameter["reference_answer"] == 0 :
                return self.rewards["rewarding_weight"] * (processed_result == 0)

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]