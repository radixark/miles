import math
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class KUR_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3589
    prompt_template = \
r"""You are given a binary string C of length {N}, defined as C[0], C[1], ..., C[{N_minus_1}].
For each index `i` (0 ≤ i < {N}):
- C[i] = 0 if and only if ({A} × i + {B}) mod {N} < {P}. It is guaranteed that {A} and {N} are coprime.
- Otherwise, C[i] = 1.

Please output how many times the following binary string appears (as a contiguous substring) in the string C: {T}"""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the KUR_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 8, "MAX_N should be greater than or equal to 8"

        assert "MAX_M" in self.parameter, "MAX_M is required in parameter"
        MAX_M = self.parameter["MAX_M"]
        assert MAX_M >= 2, "MAX_M should be greater than or equal to 2"

        while True :
            N = self.parameter["N"] = random.randint(8, MAX_N)
            A, B, P = self.parameter["A"], self.parameter["B"], self.parameter["P"] = random.randint(2, N - 1), random.randint(0, N - 1), random.randint(1, N - 1)
            if math.gcd(N, A) == 1 :
                break
        
        def compute_answer(T : str) -> int :
            M = len(T)
            intervals = []
            for x, ch in enumerate(T):
                ax = (A * x) % N
                if ch == '0':
                    l = (P - ax - B) % N
                    r = (N - ax - B) % N
                else:
                    l = (-ax - B) % N
                    r = (P - ax - B) % N
                # now l, r are in [0, N-1]
                if l <= r:
                    intervals.append((l, r - 1))
                else:
                    intervals.append((0, r - 1))
                    intervals.append((l, N - 1))

            # account for the tail positions
            for i in range(N - M + 1, N):
                intervals.append(( (A * i) % N, (A * i) % N ))

            intervals.sort()
            ans = N
            mx = -1

            for l, r in intervals:
                if l <= mx:
                    # overlapping or contiguous with previous
                    removed = max(0, r - mx)
                    ans -= removed
                    mx = max(mx, r)
                else:
                    # disjoint interval
                    ans -= (r - l + 1)
                    mx = r

            return ans

        start_i = random.randint(0, N - 2)
        T = ""
        Answer2Ts = {}
        for i in range(start_i, min(N, start_i + MAX_M)) :
            T += "0" if (A * i + B) % N < P else "1"
            answer = compute_answer(T)
            assert answer >= 1, "Answer should be at least 1"
            if answer not in Answer2Ts :
                Answer2Ts[answer] = []
            Answer2Ts[answer].append(T)
        
        self.parameter["reference_answer"] = random.choice(list(Answer2Ts.keys()))
        self.parameter["T"] = random.choice(Answer2Ts[self.parameter["reference_answer"]])
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            A = self.parameter["A"],
            B = self.parameter["B"],
            P = self.parameter["P"],
            T = self.parameter["T"],
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
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]