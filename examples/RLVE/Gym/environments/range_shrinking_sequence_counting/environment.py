import random
from array import array
from typing import Optional
from Gym.environment import VerifiableEnvironment


class RangeShrinkingSequenceCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4063
    prompt_template = \
r"""Count the number of sequences A[1], A[2], ..., A[{N}] such that:
- For each i (1 ≤ i ≤ {N}), 1 ≤ A[i] ≤ R[i], where R is given as: {R}
- For each i (3 ≤ i ≤ {N}):
  - Let r = the minimum value among A[1], ..., A[i−2] that is ≥ A[i−1] (if none exists, r = +∞).
  - Let l = the maximum value among A[1], ..., A[i−2] that is ≤ A[i−1] (if none exists, l = −∞).
  - Then A[i] must satisfy l ≤ A[i] ≤ r.

Can you let me know the number of valid sequences modulo {MOD}?"""

    def __init__(self,
                 max_MOD : int = 1000000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the RangeShrinkingSequenceCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_MOD = max_MOD
        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        shrinking_sequence = [random.randint(1, N), random.randint(1, N)]
        l, r = 1, N
        if shrinking_sequence[0] >= shrinking_sequence[1] :
            r = shrinking_sequence[1]
        if shrinking_sequence[0] <= shrinking_sequence[1] :
            l = shrinking_sequence[1]
        for i in range(2, N) :
            shrinking_sequence.append(random.randint(l, r))
            if shrinking_sequence[i - 1] >= shrinking_sequence[i] :
                assert shrinking_sequence[i - 1] <= r
                r = shrinking_sequence[i]
            if shrinking_sequence[i - 1] <= shrinking_sequence[i] :
                assert shrinking_sequence[i - 1] >= l
                l = shrinking_sequence[i]
            assert 1 <= l <= r <= N
        self.parameter["R"] = R = [random.randint(a, N) for a in shrinking_sequence]

        MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)


        MAXV = max(R) if R else 0
        SENT = MAXV + 1                # sentinel "+inf" equivalent used in the C++ code (151 there)
        SIZE = SENT + 2                # +1 for 1-based shift, +1 so we can safely index r+2 etc.
        TOT = SIZE * SIZE * SIZE

        # Helper to compute flattened 3D index with a 1-based shift on each axis.
        # We store f[L+1][R+1][x+1] at flat index ((L1*SIZE + R1) * SIZE + X1)
        def base_idx(L1, R1):
            return (L1 * SIZE + R1) * SIZE

        # Modular add/sub on array('I') cells (values kept in [0, MOD))
        def add_at(A, idx, val):
            s = A[idx] + val
            if s >= MOD:
                s -= MOD
            A[idx] = s

        def sub_at(A, idx, val):
            cur = A[idx]
            if cur >= val:
                A[idx] = cur - val
            else:
                A[idx] = cur - val + MOD

        # DP arrays as flat typed arrays (memory efficient vs nested Python lists)
        f = array('I', [0]) * TOT
        g = array('I', [0]) * TOT

        # Initialization: for i in 1..R[0], f[0][SENT][i] = 1  (shifted indices)
        L0 = 0
        Rinf = SENT
        L1 = L0 + 1
        R1 = Rinf + 1
        b = base_idx(L1, R1)
        for x in range(1, R[0] + 1):
            X1 = x + 1
            f[b + X1] = 1

        # Iterate positions 2..N
        for i in range(1, N):  # Python 0-based: position i corresponds to a[i], so start from index 1
            Ai = R[i]
            # reset g to zeros
            g = array('I', [0]) * TOT

            # transitions
            for L in range(0, SENT + 1):
                L1 = L + 1
                for RR in range(L, SENT + 1):
                    R1 = RR + 1
                    bf = base_idx(L1, R1)
                    for x in range(L, RR + 1):
                        X1 = x + 1
                        c = f[bf + X1]
                        if c == 0:
                            continue

                        # 1) choose in (L, min(x-1, Ai))
                        l = L + 1
                        r = min(x - 1, Ai)
                        if l <= r:
                            # target pair (L, x)
                            tgtL1 = L1
                            tgtR1 = X1  # since new R becomes x
                            bg = base_idx(tgtL1, tgtR1)
                            add_at(g, bg + (l + 1), c)       # + at l
                            sub_at(g, bg + (r + 1 + 1), c)   # - at r+1

                        # 2) choose in (x+1, min(RR-1, Ai))
                        l = x + 1
                        r = min(RR - 1, Ai)
                        if l <= r:
                            # target pair (x, RR)
                            tgtL1 = X1
                            tgtR1 = R1
                            bg = base_idx(tgtL1, tgtR1)
                            add_at(g, bg + (l + 1), c)
                            sub_at(g, bg + (r + 1 + 1), c)

                        # 3) choose L exactly if valid (L > 0 and L <= Ai)
                        if L != 0 and L <= Ai:
                            tgtL1 = L1
                            tgtR1 = L1
                            bg = base_idx(tgtL1, tgtR1)
                            add_at(g, bg + (L + 1), c)       # at position L
                            sub_at(g, bg + (L + 1 + 1), c)   # at L+1

                        # 4) choose RR exactly if RR is a real bound (RR <= MAXV), RR <= Ai, and L != RR
                        if RR <= Ai and RR <= MAXV and L != RR:
                            tgtL1 = R1
                            tgtR1 = R1
                            bg = base_idx(tgtL1, tgtR1)
                            add_at(g, bg + (RR + 1), c)
                            sub_at(g, bg + (RR + 1 + 1), c)

                        # 5) choose x exactly if x <= Ai and it's not equal to L or RR
                        if x <= Ai and L != x and RR != x:
                            tgtL1 = X1
                            tgtR1 = X1
                            bg = base_idx(tgtL1, tgtR1)
                            add_at(g, bg + (x + 1), c)
                            sub_at(g, bg + (x + 1 + 1), c)

            # prefix sums along the 3rd dimension: g[L][R][x] += g[L][R][x-1]
            for L in range(0, SENT + 1):
                L1 = L + 1
                for RR in range(L, SENT + 1):
                    R1 = RR + 1
                    bg = base_idx(L1, R1)
                    pref = 0
                    # x runs from L..RR, we use shifted index (x+1)
                    for x in range(L, RR + 1):
                        X1 = x + 1
                        val = g[bg + X1]
                        s = val + pref
                        if s >= MOD:
                            s -= MOD
                        g[bg + X1] = s
                        pref = s

            # f = g for next iteration
            f = g

        # Sum all f[L][R][x] over 0<=L<=R<=SENT, L<=x<=R
        ans = 0
        for L in range(0, SENT + 1):
            L1 = L + 1
            for RR in range(L, SENT + 1):
                R1 = RR + 1
                bf = base_idx(L1, R1)
                for x in range(L, RR + 1):
                    X1 = x + 1
                    val = f[bf + X1]
                    ans += val
                    if ans >= MOD:
                        ans -= MOD

        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            R = ", ".join("R[{}]={}".format(i, Ri) for i, Ri in enumerate(self.parameter["R"], start =1 )),
            MOD = self.parameter["MOD"],
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
            if not (0 <= processed_result < self.parameter["MOD"]) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]