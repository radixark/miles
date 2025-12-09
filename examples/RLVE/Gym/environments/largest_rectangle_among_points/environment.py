import random
from typing import Optional, Tuple, List
from Gym.environment import VerifiableEnvironment


class LargestRectangle_AmongPoints_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3217
    prompt_template = \
r"""You are given a set of {N} points in a 2D plane, each represented by its coordinates `(x, y)`:
{points}

Your task is to find four **distinct** points such that they form a rectangle (NOT necessarily axis-aligned). Among all such rectangles, choose one with the **maximum possible area**.

**Output Format:** Output one line containing the indices (0-based) of the four selected points, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the LargestRectangle_AmongPoints_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None:
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 5, "N should be greater than or equal to 5"

        points = self.parameter["points"] = []
        points.append((random.randint(-N // 2, +N // 2), random.randint(-N // 2, +N // 2)))
        while True :
            dx, dy = random.randint(-N // 2, +N // 2), random.randint(-N // 2, +N // 2)
            if dx == 0 and dy == 0 :
                continue
            x, y = points[0]
            points.append((x + dx, y + dy))
            points.append((x - dy, y + dx))
            points.append((x + dx - dy, y + dy + dx))
            break
        for i in range(4, N) :
            points.append((random.randint(-N, +N), random.randint(-N, +N)))
        random.shuffle(points)


        # Build list of all point‐pairs (diagonals), storing:
        # (squared_length, sum_x, sum_y, idx1, idx2)
        lines = []
        for i in range(N):
            xi, yi = points[i]
            for j in range(i + 1, N):
                xj, yj = points[j]
                dx = xi - xj
                dy = yi - yj
                s = dx * dx + dy * dy
                # midpoint * 2 is (xi+xj, yi+yj)
                sx = xi + xj
                sy = yi + yj
                lines.append((s, sx, sy, i, j))

        # Sort by (length, midpoint_x, midpoint_y)
        lines.sort(key=lambda t: (t[0], t[1], t[2]))

        ans = 0
        M = len(lines)
        # Scan through sorted diagonals, grouping by equal (s, sx, sy)
        i = 0
        while i < M:
            s0, sx0, sy0, idx1, idx2 = lines[i]
            j = i + 1
            # For each other diagonal with same length and midpoint...
            while j < M and lines[j][0] == s0 and lines[j][1] == sx0 and lines[j][2] == sy0:
                _, _, _, idx3, _ = lines[j]
                # Compute the rectangle area via the cross‐product trick:
                # area = |(C−A) × (B−A)|, with A=points[idx1], C=points[idx2], B=points[idx3]
                x1, y1 = points[idx1]  # A
                x2, y2 = points[idx2]  # C (opposite of A)
                x3, y3 = points[idx3]  # B (one endpoint of other diagonal)
                # Determinant = x1*y2 + x2*y3 + x3*y1 - x2*y1 - x3*y2 - x1*y3
                tmp = abs(x1*y2 + x2*y3 + x3*y1 - x2*y1 - x3*y2 - x1*y3)
                if tmp > ans:
                    ans = tmp
                j += 1
            i += 1

        assert ans > 0, "The maximum area should be greater than 0"
        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            points = "\n".join("Point {}: ({}, {})".format(i, x, y) for i, (x, y) in enumerate(self.parameter["points"])),
        )


    def _process(self, answer: Optional[str]) -> Optional[Tuple[int, int, int, int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                indices = list(map(int, answer.split()))
                if len(indices) != 4 :
                    return None  # Invalid answer format
                return indices[0], indices[1], indices[2], indices[3]
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output: str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None:
            assert isinstance(processed_result, tuple), "processed_result should be a tuple of indices"

            if not all(0 <= idx < self.parameter["N"] for idx in processed_result) :
                return self.rewards["invalid_solution"]

            def rectangle_area(P: List[Tuple[int, int]]) -> Optional[int]:
                A = P[0]
                others = P[1:]

                d2 = []
                for X in others:
                    dx, dy = X[0] - A[0], X[1] - A[1]
                    d2.append((dx*dx + dy*dy, X, dx, dy))
                d2.sort(key=lambda t: t[0])

                d1, B, dx1, dy1 = d2[0]
                d2_val, D, dx2, dy2 = d2[1]
                C = d2[2][1]

                # Critical fix: Check for zero-length sides (duplicate points)
                if d1 == 0 or d2_val == 0:
                    return None

                if dx1*dx2 + dy1*dy2 != 0:  # Perpendicular check
                    return None

                expected_C = (B[0] + D[0] - A[0], B[1] + D[1] - A[1])
                if expected_C != C:  # Parallelogram property
                    return None

                area = abs(dx1*dy2 - dy1*dx2)
                return area

            answer, gold = rectangle_area([self.parameter["points"][idx] for idx in processed_result]), self.parameter["gold_answer"]
            if answer is None :
                return self.rewards["invalid_solution"]
            assert answer <= gold, "The answer area should be less than or equal to the gold area"

            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else:
            return self.rewards["wrong_format"]