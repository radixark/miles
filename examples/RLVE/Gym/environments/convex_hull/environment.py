import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class ConvexHull_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a set of {N} points on a 2D plane labeled from 0 to {N_minus_1}.
It is guaranteed that:
(1) all the coordinates are integers;
(2) no two points have the same coordinates;
(3) no three points are on the same line.
Below is the set of points:
{points}

Your task is to find the **convex hull** of these points, which is the smallest convex polygon that contains all the points. 

**Output Format:** Your output should be one single **integer**, representing the value of 2 times the area of the convex hull (which can be proven to be an integer)."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the ConvexHull_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
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

        labels = list(range(len(self.parameter["points"])))
        sorted_point_labels = sorted(labels, key=lambda i: (self.parameter["points"][i][0], self.parameter["points"][i][1]))

        # calculate the convex hull using Andrew's algorithm
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for i in sorted_point_labels:
            while len(lower) >= 2 and cross_product(self.parameter["points"][lower[-2]], self.parameter["points"][lower[-1]], self.parameter["points"][i]) <= 0:
                lower.pop()
            lower.append(i)
        
        upper = []
        for i in reversed(sorted_point_labels):
            while len(upper) >= 2 and cross_product(self.parameter["points"][upper[-2]], self.parameter["points"][upper[-1]], self.parameter["points"][i]) <= 0:
                upper.pop()
            upper.append(i)
        
        convex_hull = lower[:-1] + upper[:-1]
        area = 0

        for i in range(len(convex_hull)):
            j = (i + 1) % len(convex_hull)
            x1, y1 = self.parameter["points"][convex_hull[i]]
            x2, y2 = self.parameter["points"][convex_hull[j]]
            area += x1 * y2 - x2 * y1
        
        self.parameter["reference_answer"] = abs(area)
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            points = "\n".join("({}, {})".format(x, y) for x, y in self.parameter["points"]),
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
            if processed_result <= 0 :
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