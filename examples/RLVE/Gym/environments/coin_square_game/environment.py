import random
from array import array
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class CoinSquareGame_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2964
    prompt_template = \
r"""You are given {N} coins in a row (1-indexed from left to right). The i-th coin has value C[i]: {C}
Alice and Bob play alternately, with Alice going first. On a turn, a player removes some **positive number** of **leftmost** coins and adds the sum of their values to their own score. The game ends when no coins remain.

Rules:
- On Alice’s **first** turn, she may take either 1 or 2 coins.
- Thereafter, if the previous player took k coins, the current player may take any number of coins from 1 to min(k * 2, the number of remaining coins).

Assuming both players play optimally, what is the **maximum total value** Alice can obtain?"""

    def __init__(self,
                 weight_multiple : int = 2,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the CoinSquareGame_Environment instance.
        """
        super().__init__(**kwargs)

        self.weight_multiple = weight_multiple
        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 5, "N should be greater than or equal to 5"

        C = self.parameter["C"] = [random.randint(1, N * self.weight_multiple) for _ in range(N)]

        
        A = C
        # Build prefix sums of the reversed sequence (to match the C++ approach)
        S = [0] * (N + 1)
        for i in range(1, N + 1):
            S[i] = S[i - 1] + A[N - i]

        # dp_rows[i] will store dp[i][j] for j = 0..floor((i+1)/2)
        # (indices beyond this plateau to the same value, so we clamp when reading)
        dp_rows = [None] * (N + 1)
        dp_rows[0] = array('I', [0])

        for i in range(1, N + 1):
            max_j = (i + 1) // 2
            row = array('I', [0] * (max_j + 1))
            for j in range(1, max_j + 1):
                k = 2 * j - 1
                # Start with dp[i][j-1]
                best = row[j - 1]

                # Option 1: take k coins if possible
                r = i - k
                if r >= 0:
                    prev_row = dp_rows[r]
                    prev_max_j = len(prev_row) - 1
                    idx = k if k <= prev_max_j else prev_max_j  # clamp
                    cand = S[i] - prev_row[idx]
                    if cand > best:
                        best = cand

                # Option 2: take k+1 coins if possible
                r2 = i - (k + 1)
                if r2 >= 0:
                    prev_row2 = dp_rows[r2]
                    prev2_max_j = len(prev_row2) - 1
                    idx2 = (k + 1) if (k + 1) <= prev2_max_j else prev2_max_j  # clamp
                    cand2 = S[i] - prev_row2[idx2]
                    if cand2 > best:
                        best = cand2

                row[j] = best

            dp_rows[i] = row

        self.parameter["reference_answer"] = dp_rows[N][1]
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            C = " ".join("C[{}]={}".format(i, Ci) for i, Ci in enumerate(self.parameter["C"], start = 1)),
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