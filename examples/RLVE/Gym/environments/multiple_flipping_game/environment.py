import math
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MultipleFlippingGame_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3179
    prompt_template = \
r"""You are given an array of length {N}, indexed from `1` to `{N}`.

Two players, Alice and Bob, play the following game:
+ Initially, some positions in the array are **white**, and the rest are **black**.
+ The players take turns. On each turn, the current player selects a **white** cell with index `x`.
+ Then, they choose an integer `k` such that 1 <= k <= n / x, and **flip the color** of all cells at indices `x, 2×x, ..., k×x`.
+ A player **loses** if they have no valid move on their turn.

Initially, the cells at indices {white_indices} are white (all others are black). Determine whether the **first player (Alice)** has a **winning strategy** if both players play optimally.

**Output Format:** Your final answer should be either `Yes` or `No` (do **NOT** include quotes or backticks), indicating whether the first player has a forced win."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MultipleFlippingGame_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 1, "N should be greater than or equal to 1"

        self.parameter["reference_answer"] = "Yes" if random.random() < 0.5 else "No"

        
        sn = int(math.isqrt(N))
        p = []
        r_list = []
        l = 1
        while l <= N :
            k = N // l
            r = N // k
            p.append(l)
            r_list.append(r)
            l = r + 1

        m = len(p)
        sg_small = [0] * (sn + 1)
        sg_large = [0] * (sn + 1)
        vis = [0] * (2 * sn + 5)

        for i in range(m - 1, -1, -1) :
            li = p[i]
            t = N // li
            s = 0
            l2 = 2
            mark = i + 1
            while l2 <= t :
                k2 = t // l2
                r2 = t // k2
                v = l2 * li
                if v <= sn :
                    gv = sg_small[v]
                else :
                    gv = sg_large[k2]
                vis[s ^ gv] = mark
                if ((r2 - l2 + 1) & 1) :
                    s ^= gv
                l2 = r2 + 1
            g = 1
            while vis[g] == mark :
                g += 1
            if li <= sn :
                sg_small[li] = g
            else:
                sg_large[t] = g

        def SG(x) :
            if x <= sn:
                return sg_small[x]
            return sg_large[N // x]
        

        while True :
            white_index_number = random.randint(1, N)
            white_indices = random.sample(range(1, N + 1), white_index_number)
            xo = 0
            for x in white_indices :
                xo ^= SG(x)
            if ("Yes" if xo else "No") == self.parameter["reference_answer"] :
                self.parameter["white_indices"] = white_indices
                break
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            white_indices = ", ".join(map(str, sorted(self.parameter["white_indices"]))),
        )


    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            return answer.strip()
        else :
            return None
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result not in ("Yes", "No") :
                return self.rewards["invalid_answer"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]