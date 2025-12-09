import random
import functools
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SumTriangleArea_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3476
    prompt_template = \
r"""There are {N} points in a 2D plane, each represented by its coordinates (x, y). The points are given as follows:
{points}

Please compute the **sum of the areas of all triangles** that can be formed by any three distinct points in this set. If a triangle is degenerate (i.e., the three points are collinear), its area is considered 0. **Output the total area multiplied by 2** (i.e., twice the sum of all triangle areas), which will always be an integer (think about why this is the case)."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SumTriangleArea_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        points = self.parameter["points"] = random.sample([(x, y) for x in range(0, N + 1) for y in range(0, N + 1)], N)


        A = sorted(points, key=lambda p: (p[0], p[1]))

        ans = 0
        for i in range(N):
            xi, yi = A[i]
            # build vectors from A[i] to all later points
            s = [(x - xi, y - yi) for x, y in A[i+1:]]
            # sort by polar angle around the origin using cross-product comparator
            s.sort(key=functools.cmp_to_key(
                lambda a, b: -1 if a[1]*b[0] < a[0]*b[1]
                            else (1 if a[1]*b[0] > a[0]*b[1] else 0)
            ))

            m = len(s)
            # build suffix sums of x- and y-components
            sx = [0] * (m + 1)
            sy = [0] * (m + 1)
            for j in range(m - 1, -1, -1):
                sx[j] = sx[j+1] + s[j][0]
                sy[j] = sy[j+1] + s[j][1]
                # accumulate cross-products to sum triangle areas (twice the area)
                ans += s[j][0] * sy[j+1] - s[j][1] * sx[j+1]
            
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
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
            if processed_result < 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                if processed_result == 0 :
                    return self.rewards["rewarding_weight"] * int(self.parameter["reference_answer"] == 0)
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]