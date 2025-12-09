import random
from typing import Optional
from bisect import bisect_left
from Gym.environment import VerifiableEnvironment


class SingingGirlStory_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4229
    prompt_template = \
r"""Consider an array H[1], H[2], ..., H[{N}], where each H[i] is an integer in [1, {A}]. We say max(H[l : r + 1]) denotes the maximum value in the subarray H[l], H[l+1], ..., H[r] (1 ≤ l ≤ r ≤ {N}). How many arrays H satisfy all of the following conditions?
{conditions}

Output the number of valid arrays modulo {MOD}."""
    MODs = (666623333, 998244353, 10 ** 9 + 7)

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the SingingGirlStory_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 3, "MAX_N_M should be greater than or equal to 3"

        N = self.parameter["N"] = random.randint(3, MAX_N_M)
        A = self.parameter["A"] = random.randint(2, N)
        H = [random.randint(1, A) for i in range(N)]
        M = random.randint(1, MAX_N_M)

        conditions = self.parameter["conditions"] = []
        for _ in range(M) :
            length = random.randint(2, N)
            start = random.randint(1, N - length + 1)
            end = start + length - 1
            conditions.append((start, end, max(H[start - 1 : (end - 1) + 1])))
            assert 1 <= conditions[-1][0] <= conditions[-1][1] <= N, "1 <= l <= r <= N"
            assert 1 <= conditions[-1][2] <= A, "max(H[l : r + 1]) should be in [1, A]"
        
        MOD = self.parameter["MOD"] = random.choice(self.MODs)


        def calc(val, pts, eves, UNI, Q):
            # pts: list of segment indices i (1-based) where mx[i] == val
            # eves: list of event indices (1-based) where Q[id]['v'] == val
            if not pts:
                return 0
            L = len(pts)
            # 1-based for convenience; Aindex[0] = 0 as in the C++ code
            Aindex = [0] + pts[:]

            # Precompute powers
            PPW = [1] * (L + 1)  # PPW[0] = 1 is safe
            for i in range(1, L + 1):
                seg_len = UNI[Aindex[i] + 1] - UNI[Aindex[i]]
                PPW[i] = pow(val - 1, seg_len, MOD)

            DP = [0] * (L + 1)
            DP[0] = 1

            for i in range(1, L + 1):
                seg_len = UNI[Aindex[i] + 1] - UNI[Aindex[i]]
                pw = (pow(val, seg_len, MOD) - pow(val - 1, seg_len, MOD) + MOD) % MOD
                mxL = 0
                for eid in eves:
                    if Q[eid]['r'] <= Aindex[i]:
                        if Q[eid]['l'] > mxL:
                            mxL = Q[eid]['l']
                j = i - 1
                while j >= 0 and Aindex[j] >= mxL:
                    DP[i] = (DP[i] + DP[j] * pw) % MOD
                    pw = (pw * PPW[j]) % MOD
                    j -= 1

            res = 0
            for i in range(0, L + 1):
                ok = True
                for eid in eves:
                    if Q[eid]['l'] > Aindex[i]:
                        ok = False
                        break
                if ok:
                    pw = 1
                    for j in range(i + 1, L + 1):
                        pw = (pw * PPW[j]) % MOD
                    res = (res + DP[i] * pw) % MOD
            return res

        def solve_one():
            # Read queries
            Q = [None] * (M + 1)  # 1-based
            KEY = []
            ST = set()
            for i, (l, r, v) in enumerate(conditions, start = 1):
                r += 1
                Q[i] = {'l': l, 'r': r, 'v': v}
                KEY.append(l)
                KEY.append(r)
                ST.add(v)

            # Coordinate compression for boundaries
            KEY.sort()
            UNI = [None]  # 1-based
            prev = None
            for x in KEY:
                if x != prev:
                    UNI.append(x)
                    prev = x
            NUM = len(UNI) - 1  # number of unique keys
            UNI.append(N + 1)   # uni[NUM+1] = N+1

            # Map l, r to indices in UNI[1..NUM]
            for i in range(1, M + 1):
                lval = Q[i]['l']
                rval = Q[i]['r']
                li = bisect_left(UNI, lval, 1, NUM + 1)
                ri = bisect_left(UNI, rval, 1, NUM + 1)
                Q[i]['l'] = li
                Q[i]['r'] = ri

            # Compute per-segment minimal mx (INF if unconstrained)
            INF = A + 1  # computed based on input
            MX = [INF] * (NUM + 2)  # 1-based up to NUM
            for i in range(1, M + 1):
                for j in range(Q[i]['l'], Q[i]['r']):
                    if Q[i]['v'] < MX[j]:
                        MX[j] = Q[i]['v']

            # Sum of constrained lengths
            total_constrained = 0
            for i in range(1, NUM + 1):
                if MX[i] != INF:
                    total_constrained += (UNI[i + 1] - UNI[i])

            prd = pow(A, N - total_constrained, MOD)

            # Multiply contributions for each distinct maximum value
            for val in ST:
                pts = [i for i in range(1, NUM + 1) if MX[i] == val]
                eves = [i for i in range(1, M + 1) if Q[i]['v'] == val]
                prd = (prd * calc(val, pts, eves, UNI, Q)) % MOD

            return prd

        self.parameter["reference_answer"] = solve_one()
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = self.parameter["A"],
            conditions = "\n".join("- max(H[{} : {} + 1]) = {}".format(l, r, v) for (l, r, v) in self.parameter["conditions"]),
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