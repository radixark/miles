import random
from collections import deque
from typing import Optional, Tuple
from Gym.environment import VerifiableEnvironment


class WIL_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3594
    prompt_template = \
r"""You are given an array `A` of length {N}, indexed from 1 to {N}. The array is: {A}

Your task is as follows:
1. First, choose an interval [l1, r1] (such that r1 - l1 + 1 <= {D}) and set all A[i] = 0 for l1 ≤ i ≤ r1.
2. Then, find an interval [l2, r2] such that the **sum** of A[i] over l2 ≤ i ≤ r2 is at most {P}, and the **length** of this interval is as long as possible.

Output `l1`, `r1`, `l2`, and `r2` (in order) — separated by spaces in a single line."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the WIL_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        A = self.parameter["A"] = [random.randint(1, N) for _ in range(N)]
        D = self.parameter["D"] = random.randint(1, N - 1)
        P = self.parameter["P"] = random.randint(1, sum(A) - sum(sorted(A, reverse = True)[: D]))


        # Build prefix sums S where S[i] = sum of A[0..i-1]
        S = [0] * (N + 1)
        for i in range(1, N + 1):
            S[i] = S[i - 1] + A[i - 1]
        
        # Deque to maintain candidate segment endpoints (indices in [D..N])
        # sorted so that the front q[0] has the segment of length D with the largest sum
        q = deque([D])
        
        ans = D     # we can always zero out one segment of length D, giving at least length D
        l = 1       # current window left endpoint (1-based for S)
        
        # Slide right endpoint i from D+1 to N (1-based)
        for i in range(D + 1, N + 1):
            # Add the new segment [i-D+1..i], with sum = S[i] - S[i-D].
            # Maintain deque in decreasing order of segment-sums.
            curr_seg_sum = S[i] - S[i - D]
            while q and curr_seg_sum > (S[q[-1]] - S[q[-1] - D]):
                q.pop()
            q.append(i)
            
            # Move l forward while the best window [l..i] (minus best segment) exceeds P
            # Best segment to zero is the one at q[0]
            while q and S[i] - S[l - 1] - (S[q[0]] - S[q[0] - D]) > P:
                l += 1
                # Drop any segments that no longer fit entirely in [l..i]
                while q and (q[0] - D + 1) < l:
                    q.popleft()
            
            # Update answer: window length is i - l + 1
            ans = max(ans, i - l + 1)
        
        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = ", ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"], start = 1)),
            D = self.parameter["D"],
            P = self.parameter["P"],
        )


    def _process(self, answer : Optional[str]) -> Optional[Tuple[int, int, int, int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                l1, r1, l2, r2 = map(int, answer.split())
                return l1, r1, l2, r2
            except :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            l1, r1, l2, r2 = processed_result
            if not (1 <= l1 <= r1 <= self.parameter["N"] and 1 <= l2 <= r2 <= self.parameter["N"]) :
                return self.rewards["invalid_solution"]
            
            if r1 - l1 + 1 > self.parameter["D"] :
                return self.rewards["invalid_solution"]

            A = self.parameter["A"].copy()
            for i in range(l1, r1 + 1) :
                A[i - 1] = 0
            if sum(A[l2 - 1 : r2]) > self.parameter["P"] :
                return self.rewards["invalid_solution"]
            
            answer, gold = r2 - l2 + 1, self.parameter["gold_answer"]
            assert 0 < answer <= gold, "Answer length should not exceed gold length"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * int(answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]