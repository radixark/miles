import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class StoneIntervalsGame_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3235
    prompt_template = \
r"""There are {N} piles of stones. Initially, the i-th pile contains A[i] stones, given as: {A}
Alice and Bob play a game with the following rules:
- Alice goes first. They alternate turns.
- On each turn, a player selects a pile `i` such that **at least one of its adjacent piles** (`i - 1` or `i + 1`, if within bounds) contains **0 stones** (noting that the first/last pile has ONLY ONE adjacent pile). The player then collects **all stones** from pile `i` (pile `i` becomes 0).
- The game ends when there are no piles with any stones remaining.

Assuming both players play optimally to maximize their own total number of collected stones, output the number of stones Alice will collect."""
    
    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the StoneIntervalsGame_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        A = self.parameter["A"] = [random.randint(1, N * 2) for _ in range(N)]
        for zero_index in random.sample(range(N), random.randint(1, N - 2)) :
            A[zero_index] = 0
        

        v = A.copy()
        SumVal = sum(v)

        # mark which piles are non-zero
        tag = [x != 0 for x in v]

        # doubly-linked list over 0..N-1
        prev_ = [i - 1 for i in range(N)]
        next_ = [i + 1 for i in range(N)]
        prev_[0] = None
        next_[N - 1] = None

        head = 0
        tail = N - 1

        # 1) Triple-compression: whenever three consecutive non-zero piles
        #    form a “peak” (middle ≥ both neighbors), merge them into the rightmost.
        i = head
        while i is not None:
            while (
                prev_[i] is not None
                and prev_[prev_[i]] is not None
                and tag[i]
                and tag[prev_[i]]
                and tag[prev_[prev_[i]]]
                and v[prev_[i]] >= v[prev_[prev_[i]]]
                and v[prev_[i]] >= v[i]
            ):
                p = prev_[i]
                pp = prev_[p]
                new_prev = prev_[pp]
                # merge: v[i] = v[pp] + v[i] − v[p]
                v[i] = v[pp] + v[i] - v[p]
                # remove pp and p by re-linking new_prev ↔ i
                prev_[i] = new_prev
                if new_prev is not None:
                    next_[new_prev] = i
                else:
                    head = i
            i = next_[i]

        # 2) Edge-peeling: greedily remove matching monotonic pairs at the ends,
        #    accumulating their difference into S
        L, R = head, tail
        S = 0
        # left side
        while True:
            nl = next_[L]
            if nl is None or not (tag[L] and tag[nl]) or v[L] < v[nl]:
                break
            S += v[nl] - v[L]
            L = next_[nl]
        # right side
        while True:
            pr = prev_[R]
            if pr is None or not (tag[R] and tag[pr]) or v[R] < v[pr]:
                break
            S += v[pr] - v[R]
            R = prev_[pr]

        # 3) Collect the remaining non-zero segments between L and R
        segments = []
        i = L
        while True:
            if tag[i]:
                segments.append(v[i])
            if i == R:
                break
            i = next_[i]

        # 4) Sort descending, append the peeled sum S, then do an alternating sum
        segments.sort(reverse=True)
        segments.append(S)
        score = 0
        for idx, val in enumerate(segments):
            score += val if idx % 2 == 0 else -val

        # 5) Recover each player's total
        self.parameter["reference_answer"] = (SumVal + score) // 2
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
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