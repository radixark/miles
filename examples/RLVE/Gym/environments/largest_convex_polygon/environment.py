import random
from functools import cmp_to_key
from typing import Optional, List, Tuple
from Gym.environment import VerifiableEnvironment


class LargestConvexPolygon_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2924
    prompt_template = \
r"""You are given {N} points in the 2D plane, labeled from 1 to {N}. No two points share the same coordinates, and no three points are collinear:
{points}

Find a subset of distinct points that forms the vertices of a **convex polygon**, and maximize the number of points in this subset; please output the labels of the selected points in one line, separated by spaces (in any order); if multiple answers exist, output any one."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the LargestConvexPolygon_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "unsuccessful_solution" : unsuccessful_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None:
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        self.parameter["points"] = set()
        lines = set()
        for i in range(N):
            while True:
                x = random.randint(0, N)
                y = random.randint(0, N)
                if (x, y) in self.parameter["points"]:
                    continue

                coline = False
                new_lines = set()
                for (px, py) in self.parameter["points"]:
                    if px == x:
                        a, b, c = 1, 0, -x
                    else:
                        a, b = py - y, x - px
                        c = -(a * x + b * y)
                    
                    def gcd(a, b):
                        while b:
                            a, b = b, a % b
                        return a
                    
                    g = gcd(abs(a), gcd(abs(b), abs(c)))
                    a, b, c = a // g, b // g, c // g

                    if a < 0:
                        a, b, c = -a, -b, -c
                    elif a == 0 and b < 0:
                        b, c = -b, -c
                    
                    if (a, b, c) in lines:
                        coline = True
                        break
                    
                    new_lines.add((a, b, c))

                if coline:
                    continue

                self.parameter["points"].add((x, y))
                lines.update(new_lines)
                break
        
        self.parameter["points"] = list(self.parameter["points"])


        P = self.parameter["points"]

        def octant(dx, dy):
            if dx == 0 and dy > 0:   # up
                return 1
            elif dx > 0 and dy > 0:  # NE
                return 2
            elif dx > 0 and dy == 0: # right
                return 3
            elif dx > 0 and dy < 0:  # SE
                return 4
            elif dx == 0 and dy < 0: # down
                return 5
            elif dx < 0 and dy < 0:  # SW
                return 6
            elif dx < 0 and dy == 0: # left
                return 7
            else:                    # dx < 0 and dy > 0 -> NW
                return 8

        # Build all directed edges with precomputed (dx, dy, oct)
        edges = []
        for u in range(N):
            xu, yu = P[u]
            for v in range(N):
                if u == v:
                    continue
                xv, yv = P[v]
                dx = xv - xu
                dy = yv - yu
                edges.append((u, v, dx, dy, octant(dx, dy)))

        def cmp_edges(e1, e2):
            # sort by octant first (clockwise starting from up),
            # then by slope via cross product (dy1*dx2 ? dy2*dx1)
            if e1[4] != e2[4]:
                return -1 if e1[4] < e2[4] else 1
            cross = e1[3] * e2[2] - e2[3] * e1[2]  # dy1*dx2 - dy2*dx1
            if cross > 0:
                return -1
            elif cross < 0:
                return 1
            else:
                return 0

        edges.sort(key=cmp_to_key(cmp_edges))

        # Only keep (u, v) for the DP loop
        EV = [(u, v) for (u, v, _, _, _) in edges]

        ans = 0
        for i in range(N):
            mx = [None] * N
            mx[i] = 0
            for u, v in EV:
                val = mx[u]
                if val is not None:
                    cand = val + 1
                    if mx[v] is None or cand > mx[v]:
                        mx[v] = cand
            if mx[i] is not None and mx[i] > ans:
                ans = mx[i]
        assert ans >= 3, "The answer should be greater than or equal to 3"
        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            points = "\n".join("Point {}: ({}, {})".format(i, x, y) for i, (x, y) in enumerate(self.parameter["points"], start = 1)),
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
            if not all(1 <= i <= self.parameter["N"] for i in processed_result) :
                return self.rewards["invalid_solution"]
            if len(processed_result) != len(set(processed_result)) :
                return self.rewards["invalid_solution"]

            def cross(o: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]) -> int:
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

            def can_form_convex_polygon(points: List[Tuple[int, int]]) -> bool:
                pts = sorted(set(points))
                n = len(pts)
                if n < 3:
                    return False

                lower = []
                for p in pts:
                    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                        lower.pop()
                    lower.append(p)

                upper = []
                for p in reversed(pts):
                    while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                        upper.pop()
                    upper.append(p)

                hull = lower[:-1] + upper[:-1]
                return len(hull) == n

            if not can_form_convex_polygon([self.parameter["points"][i - 1] for i in processed_result]) :
                return self.rewards["unsuccessful_solution"]

            answer, gold = len(processed_result), self.parameter["gold_answer"]
            assert answer <= gold, "The answer should be less than or equal to the gold answer"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]