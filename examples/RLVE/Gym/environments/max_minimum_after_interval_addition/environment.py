import heapq
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MaxMinimum_AfterIntervalAddition_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4064
    prompt_template = \
r"""You are given an array `ARRAY` of length {N}: {ARRAY}

You are also given {M} intervals (numbered 1 to {M}):
{intervals}

Let's select {K} **distinct** intervals; for each selected interval [l, r], add the value {A} to every element of `ARRAY` from index l to r (inclusive); all additions are cumulative. Can we **maximize the minimum value** in `ARRAY` after applying all {K} additions? You must output {K} integers in one line — the selected interval indices (in any order), separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MaxMinimum_AfterIntervalAddition_Environment instance.
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
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 3, "MAX_N_M should be greater than or equal to 3"

        N = self.parameter["N"] = random.randint(3, MAX_N_M)
        M = self.parameter["M"] = random.randint(3, MAX_N_M)
        K = self.parameter["K"] = random.randint(2, M - 1)

        A = self.parameter["A"] = random.randint(1, MAX_N_M)
        ARRAY = self.parameter["ARRAY"] = [random.randint(1, MAX_N_M * random.randint(1, K)) for _ in range(N)]

        intervals = self.parameter["intervals"] = []
        for i in range(M) :
            length = random.randint(1, N)
            start = random.randint(1, N - length + 1)
            intervals.append((start, start + length - 1))
        

        # Build operations list
        # Each op is a tuple: (pos, tp, val)
        # tp: 0 = left endpoint, 1 = sequence point, 2 = right endpoint
        OPS = []
        # sequence points
        for i in range(1, N + 1):
            t = ARRAY[i - 1]
            OPS.append((i, 1, t))

        # intervals
        # R[i] stores right endpoint of interval i (1-based)
        R = [0] * (M + 1)
        for i, (L_i, R_i) in enumerate(intervals, start = 1):
            OPS.append((L_i, 0, i))  # left endpoint event
            OPS.append((R_i, 2, i))  # right endpoint event
            R[i] = R_i

        # sort by position, and for ties: left(0) < sequence(1) < right(2)
        OPS.sort(key=lambda x: (x[0], x[1]))

        lf = min(ARRAY)  # lower bound (minimum current value)
        ri = lf + M * A  # upper bound (loose, but faithful to the C++)

        # jud(mid) checks if we can achieve min >= mid using at most K intervals
        def jud(mid: int) -> bool:
            flow = 0  # current accumulated +a from chosen intervals covering current position
            tot = 0   # total intervals chosen so far
            # priority queue (max-heap by r[v]); Python has min-heap, so push (-R[v], v)
            pq = []
            # book[v] == 1 means interval v has been selected
            book = [0] * (M + 1)

            for pos, tp, val in OPS:
                if tp == 0:
                    # insert left endpoint
                    v = val
                    heapq.heappush(pq, (-R[v], v))
                elif tp == 1:
                    # sequence point
                    ned = mid - val - flow
                    if ned < 0:
                        continue
                    ch = (ned + A - 1) // A  # ceil division
                    if tot + ch > K:
                        return False
                    while pq and ch:
                        _, v = heapq.heappop(pq)
                        if R[v] < pos:
                            return False
                        book[v] = 1
                        flow += A
                        ch -= 1
                        tot += 1
                    if ch > 0:
                        return False
                else:
                    # right endpoint; remove its contribution if it was chosen
                    v = val
                    if book[v]:
                        flow -= A
            return True

        while lf != ri:
            mid = (lf + ri + 1) // 2
            if jud(mid):
                lf = mid
            else:
                ri = mid - 1

        self.parameter["gold_answer"] = lf
        assert lf > 0, "The gold answer should be positive"
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            ARRAY = ", ".join("ARRAY[{}]={}".format(i, ARRAYi) for i, ARRAYi in enumerate(self.parameter["ARRAY"], start = 1)),
            M = self.parameter["M"],
            K = self.parameter["K"],
            intervals = "\n".join("Interval {}: [{}, {}]".format(i, Li, Ri) for i, (Li, Ri) in enumerate(self.parameter["intervals"], start = 1)),
            A = self.parameter["A"],
        )


    def _process(self, answer : Optional[str]) -> Optional[List[int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if len(processed_result) != self.parameter["K"] :
                return self.rewards["invalid_solution"]
            if len(set(processed_result)) != self.parameter["K"] :
                return self.rewards["invalid_solution"]
            if not all(1 <= idx <= self.parameter["M"] for idx in processed_result) :
                return self.rewards["invalid_solution"]
            
            ARRAY = self.parameter["ARRAY"].copy()
            for idx in processed_result :
                l, r = self.parameter["intervals"][idx - 1]
                l -= 1
                r -= 1
                for i in range(l, r + 1) :
                    ARRAY[i] += self.parameter["A"]
            answer, gold = min(ARRAY), self.parameter["gold_answer"]
            assert 0 < answer <= gold, "The answer should not exceed the gold answer"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise ValueError("Invalid rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]