import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SquareUndamagedPointCounting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Please count the number of distinct squares (not necessarily axis-aligned) such that:
- All four vertices are integer coordinate points with 0 ≤ x ≤ {N} and 0 ≤ y ≤ {M}.
- None of the four vertices is among the damaged points. The list of damaged points is given as follows: {damaged_points}"""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SquareUndamagedPointCounting problem.
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

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(1, MAX_N_M), random.randint(1, MAX_N_M)
        damaged_points = self.parameter["damaged_points"] = random.sample([(x, y) for x in range(N + 1) for y in range(M + 1)], random.randint(1, min(N * M, MAX_N_M)))


        pts = damaged_points.copy()  # copy to avoid modifying the original list
        pts.sort()                         # sort exactly as in the C++ code

        # compress each (x, y) to a single integer id = x*(M+1)+y for O(1) lookup
        deleted = {x * (M + 1) + y for (x, y) in pts}
        get_id = lambda x, y: x * (M + 1) + y

        # ---------- cnt0 : total number of squares in a complete grid ----------
        # cnt0 = Σ_{size = 1..min(N,M)} (N - size + 1)*(M - size + 1)*size
        limit = min(N, M)
        cnt0 = 0
        for s in range(1, limit + 1):
            cnt0 += (N - s + 1) * (M - s + 1) * s

        # ---------- cnt1 : squares counted by at least one deleted vertex ----------
        def add_lgh(lim: int, len1: int, len2: int) -> int:
            """exactly matches the lgh lambda in the C++ code"""
            res = lim * (lim + 3) // 2
            if lim > len1:
                d = lim - len1
                res -= d * (d + 1) // 2
            if lim > len2:
                d = lim - len2
                res -= d * (d + 1) // 2
            return res

        cnt1 = 0
        for x, y in pts:
            u, d = x, N - x             # up / down steps we can take
            l, r = y, M - y             # left / right steps we can take
            cnt1 += add_lgh(min(M, u), l, r)
            cnt1 += add_lgh(min(M, d), l, r)
            cnt1 += add_lgh(min(N, l), u, d)
            cnt1 += add_lgh(min(N, r), u, d)
            cnt1 -= min(l, u)
            cnt1 -= min(u, r)
            cnt1 -= min(r, d)
            cnt1 -= min(d, l)

        # ---------- cnt2 / cnt3 / cnt4 : inclusion–exclusion on pairs ----------
        cnt2 = cnt3 = cnt4 = 0
        Klen = len(pts)

        def inside(x: int, y: int) -> bool:
            return 0 <= x <= N and 0 <= y <= M

        def process(x3: int, y3: int, x4: int, y4: int) -> None:
            """one candidate square determined by the current pair of points"""
            nonlocal cnt2, cnt3, cnt4
            if not (inside(x3, y3) and inside(x4, y4)):
                return
            t1 = get_id(x3, y3) in deleted
            t2 = get_id(x4, y4) in deleted
            cnt2 += 1
            if t1: cnt3 += 1
            if t2: cnt3 += 1
            if t1 and t2: cnt4 += 1

        for i in range(Klen):
            x1, y1 = pts[i]
            for j in range(i + 1, Klen):
                x2, y2 = pts[j]

                # the two orientations where (x1,y1)–(x2,y2) is a side
                process(x1 - (y2 - y1), y1 + (x2 - x1),
                        x2 - (y2 - y1), y2 + (x2 - x1))
                process(x1 + (y2 - y1), y1 - (x2 - x1),
                        x2 + (y2 - y1), y2 - (x2 - x1))

                # orientation where they are the diagonal
                a = (x2 - x1) + (y2 - y1)
                b = (x2 - x1) - (y2 - y1)
                if (a & 1) or (b & 1):         # both must be even
                    continue
                a //= 2
                b //= 2
                process(x1 + b, y1 + a, x2 - b, y2 - a)

        # correct over-counting (each square appears C(3,1)=3 or C(4,2)=6 times)
        cnt3 //= 3
        cnt4 //= 6

        # ---------- final inclusion–exclusion ----------
        self.parameter["reference_answer"] = cnt0 - cnt1 + cnt2 - cnt3 + cnt4
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            damaged_points = ", ".join("({}, {})".format(x, y) for x, y in self.parameter["damaged_points"]),
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
                if processed_result == 0 :
                    return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == 0)
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]