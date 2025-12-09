import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SkaRockGarden_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3458
    prompt_template = \
r"""There are {N} points in a 2D plane, where the i-th point is (X[i], Y[i]) for 0 ≤ i < {N}. Each point has a cost M[i] to swap its coordinates (i.e., swapping (x, y) becomes (y, x)). Your goal is as follows:
- First, minimize the total perimeter of the smallest axis-aligned rectangle that can enclose all points after some of them are optionally swapped. The perimeter is obviously 2 × ((max_x - min_x) + (max_y - min_y)), where max_x and min_x are the maximum and minimum x-coordinates after your swaps (similarly for y).
- If multiple swap strategies result in the same minimum perimeter, choose the one with the smallest total swap cost (i.e., sum of M[i] for all swapped points).

X, Y, and M are given as follows:
{X_Y_M}

**Output Format:** Output a single line of {N} characters (no spaces or any other kinds of separators). The i-th character should be:
- `'0'` if you do **NOT** swap point i,
- `'1'` if you **do** swap point i."""

    def __init__(self,
                 wrong_format: float = -1.0,
                 rewarding_strategy_perimeter: str = "(gold/answer)^beta", rewarding_weight_perimeter: float = +0.5, rewarding_beta_perimeter: float = 5.0,
                 rewarding_strategy_cost: str = "(gold/answer)^beta", rewarding_weight_cost: float = +0.5, rewarding_beta_cost: float = 5.0,
                 **kwargs):
        """
        Initialize the SkaRockGarden_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "rewarding_strategy_perimeter": rewarding_strategy_perimeter,
            "rewarding_weight_perimeter": rewarding_weight_perimeter,
            "rewarding_beta_perimeter": rewarding_beta_perimeter,
            "rewarding_strategy_cost": rewarding_strategy_cost,
            "rewarding_weight_cost": rewarding_weight_cost,
            "rewarding_beta_cost": rewarding_beta_cost,
        }
    

    def _generate(self) -> None:
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        X, Y, M = self.parameter["X"], self.parameter["Y"], self.parameter["M"] = [random.randint(0, 2 * N) for _ in range(N)], [random.randint(0, 2 * N) for _ in range(N)], [random.randint(1, N) for _ in range(N)]


        INF = (max(max(X), max(Y)) + 1) * 2
        lx = INF
        rx = -INF
        ly = INF
        ry = -INF

        # Determine the minimal enclosing rectangle assuming no more swaps
        for i in range(N):
            x, y = X[i], Y[i]
            if x <= y:
                if x < lx: lx = x
                if x > rx: rx = x
                if y < ly: ly = y
                if y > ry: ry = y
            else:
                # these points are effectively swapped
                if y < lx: lx = y
                if y > rx: rx = y
                if x < ly: ly = x
                if x > ry: ry = x

        # The minimal fence length (perimeter of axis-aligned rectangle)
        fence_length = 2 * ((rx - lx) + (ry - ly))

        best_weight = sum(M)  # Start with the worst case: swap all points
        best_assign = None

        def try_bounds(lx0, rx0, ly0, ry0):
            """Try using bounds [lx0,rx0] × [ly0,ry0], returning (weight, assignment)
            or (None, None) if impossible."""
            total = 0
            assign = [0] * N
            for i in range(N):
                x, y = X[i], Y[i]
                if lx0 <= x <= rx0 and ly0 <= y <= ry0:
                    # no swap needed
                    assign[i] = 0
                elif lx0 <= y <= rx0 and ly0 <= x <= ry0:
                    # swap needed
                    assign[i] = 1
                    total += M[i]
                else:
                    # this point can't fit even if swapped
                    return None, None
            return total, assign

        # Try the 4 possible ways of interpreting the bounding box
        for (a, b, c, d) in (
            (lx, rx, ly, ry),
            (lx, ry, ly, rx),
            (ly, rx, lx, ry),
            (ly, ry, lx, rx),
        ):
            w, assn = try_bounds(a, b, c, d)
            if w is not None and w < best_weight:
                best_weight = w
                best_assign = assn

        # Output results
        self.parameter["gold_answer_perimeter"] = fence_length
        self.parameter["gold_answer_cost"] = best_weight
        self.parameter["reference_answer"] = "".join(map(str, best_assign))
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            X_Y_M = "\n".join("X[{}]={} Y[{}]={} M[{}]={}".format(i, Xi, i, Yi, i, Mi) for i, (Xi, Yi, Mi) in enumerate(zip(self.parameter["X"], self.parameter["Y"], self.parameter["M"]))),
        )


    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            answer = answer.strip()
            return answer
        else :
            return None
    
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if len(processed_result) != self.parameter["N"] :
                return self.rewards["wrong_format"]
            if not all(c in "01" for c in processed_result) :
                return self.rewards["wrong_format"]

            X, Y = self.parameter["X"].copy(), self.parameter["Y"].copy()
            answer_cost, gold_cost = 0, self.parameter["gold_answer_cost"]
            for i, swap in enumerate(processed_result) :
                if swap == "1" :
                    X[i], Y[i] = Y[i], X[i]
                    answer_cost += self.parameter["M"][i]
                elif swap == "0" :
                    continue
                else :
                    assert False
            answer_perimeter, gold_perimeter = 2 * ((max(X) - min(X)) + (max(Y) - min(Y))), self.parameter["gold_answer_perimeter"]

            reward = 0.0

            assert gold_perimeter <= answer_perimeter, "answer_perimeter should be greater than or equal to gold_perimeter"
            if self.rewards["rewarding_strategy_perimeter"] == "(gold/answer)^beta" :
                if answer_perimeter == 0 :
                    assert gold_perimeter == 0, "If answer_perimeter is zero, gold_perimeter should also be zero"
                    reward += self.rewards["rewarding_weight_perimeter"] * 1.0
                else :
                    reward += self.rewards["rewarding_weight_perimeter"] * ((gold_perimeter / answer_perimeter) ** self.rewards["rewarding_beta_perimeter"])
            elif self.rewards["rewarding_weight_perimeter"] == "gold=answer" :
                reward += self.rewards["rewarding_beta_weight"] * (gold_perimeter == answer_perimeter)
            else :
                raise NotImplementedError(f"Unknown rewarding strategy: {self.rewards['rewarding_strategy_perimeter']}")

            if gold_perimeter == answer_perimeter :
                assert gold_cost <= answer_cost, "answer_cost should be greater than or equal to gold_cost"
                if self.rewards["rewarding_strategy_cost"] == "(gold/answer)^beta" :
                    if answer_cost == 0 :
                        assert gold_cost == 0, "If answer_cost is zero, gold_cost should also be zero"
                        reward += self.rewards["rewarding_weight_cost"] * 1.0
                    else :
                        reward += self.rewards["rewarding_weight_cost"] * ((gold_cost / answer_cost) ** self.rewards["rewarding_beta_cost"])
                elif self.rewards["rewarding_strategy_cost"] == "gold=answer" :
                    reward += self.rewards["rewarding_weight_cost"] * (gold_cost == answer_cost)
                else :
                    raise NotImplementedError(f"Unknown rewarding strategy: {self.rewards['rewarding_strategy_cost']}")
                
            return reward
        else :
            return self.rewards["wrong_format"]