import random
from math import sqrt
from typing import Optional
from Gym.environment import VerifiableEnvironment

def distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def circle_from_two_points(p1, p2):
    center_x = (p1[0] + p2[0]) / 2
    center_y = (p1[1] + p2[1]) / 2
    radius = distance(p1, p2) / 2
    return (center_x, center_y), radius

def circle_from_three_points(p1, p2, p3):
    a1 = p2[0] - p1[0]
    b1 = p2[1] - p1[1]
    c1 = (a1 * a1 + b1 * b1) / 2
    a2 = p3[0] - p1[0]
    b2 = p3[1] - p1[1]
    c2 = (a2 * a2 + b2 * b2) / 2
    d = a1 * b2 - a2 * b1
    center_x = p1[0] + (c1 * b2 - c2 * b1) / d
    center_y = p1[1] + (a1 * c2 - a2 * c1) / d
    radius = distance((center_x, center_y), p1)
    return (center_x, center_y), radius

class SmallestCircle_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a set of {N} points on a 2D plane.
It is guaranteed that:
(1) all the coordinates are integers;
(2) no two points have the same coordinates;
(3) no three points are on the same line.
Below is the set of points:
{points}

Your task is to find the **smallest circle** covering these points, measured by the radius of the circle.
Your score will be based on the feasibility of your output and the optimality of the radius.
The precision tolerance is 0.001.

**Output Format:** Your output should be three **floats** in a single line, $x$, $y$, and $r$, separated by spaces.
$x$ and $y$ represent the center of the circle, and $r$ represents the radius of the circle."""
    epsilon = 1E-3

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = 0.0, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the Smallest Circle problem.
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
        assert N >= 2, "N should be greater than or equal to 2"

        self.parameter["points"] = set()
        lines = set()
        for i in range(N):
            while True:
                x = random.randint(0, 2 * N)
                y = random.randint(0, 2 * N)
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

        # use the randomized algorithm to find the smallest circle
        random.shuffle(self.parameter["points"])
        c = self.parameter["points"][0]
        r = 0.0
        for  i in range(1, N):
            if distance(self.parameter["points"][i], c) < r + self.epsilon:
                continue

            c = self.parameter["points"][i]
            r = 0.0
            for j in range(i):
                if distance(self.parameter["points"][j], c) < r + self.epsilon:
                    continue

                c, r = circle_from_two_points(
                    self.parameter["points"][i],
                    self.parameter["points"][j],
                )
                for k in range(j):
                    if distance(self.parameter["points"][k], c) < r + self.epsilon:
                        continue

                    c, r = circle_from_three_points(
                        self.parameter["points"][i],
                        self.parameter["points"][j],
                        self.parameter["points"][k],
                    )
        
        self.parameter["reference_answer"] = "{} {} {}".format(c[0], c[1], r)
        self.parameter["gold_answer"] = r

        if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
            self.passing_reward_threshold = self.rewards["rewarding_weight"] * ((r / (r + self.epsilon)) ** self.rewards["rewarding_beta"])
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            points = "\n".join("({}, {})".format(x, y) for x, y in self.parameter["points"]),
        )

    def _process(self, answer : Optional[str]) -> Optional[int] :
        if answer is not None :
            answer = answer.strip()
            try :
                x, y, r = map(float, answer.split())
                return (x, y, r)
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            x, y, r = processed_result
            if r <= 0:
                return self.rewards["wrong_format"]
            
            if any(distance((x, y), p) > r + self.epsilon for p in self.parameter["points"]):
                return self.rewards["invalid_solution"]
            
            opt_r = self.parameter["gold_answer"]
            assert r >= opt_r - 2 * self.epsilon, "The radius of the output circle should be at least as large as the optimal radius."

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * min(((opt_r / r) ** self.rewards["rewarding_beta"]), 1.0)
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (abs(r - opt_r) < self.epsilon)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]