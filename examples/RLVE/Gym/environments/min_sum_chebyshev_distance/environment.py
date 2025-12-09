import random
from typing import Optional, Tuple
from Gym.environment import VerifiableEnvironment


class MinSumChebyshevDistance_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given three arrays X, Y, and T, each of length {N}:
{X_Y_T}

Please find an integer point (x, y) such that the following sum is minimized: sum over 0 <= i < {N} of max(|x - X[i]|, |y - Y[i]|) * T[i]. Output a single line containing two integers x and y (separated by a space), representing the optimal point."""

    def __init__(self,
                 wrong_format : float = -1.0,rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinSumChebyshevDistance_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        X, Y, T = self.parameter["X"], self.parameter["Y"], self.parameter["T"] = [random.randint(1, 2 * N) for _ in range(N)], [random.randint(1, 2 * N) for _ in range(N)], [random.randint(1, N) for _ in range(N)]


        # A and B for rotated coordinates, C for original points
        A = []  # list of [x_rot, count]
        B = []  # list of [y_rot, count]
        C = []  # list of (u, v, count)

        for u, v, t in zip(X, Y, T):
            x_rot = u + v
            y_rot = u - v
            A.append([x_rot, t])
            B.append([y_rot, t])
            C.append((u, v, t))

        # Sort by rotated coordinates
        A.sort(key=lambda item: item[0])
        B.sort(key=lambda item: item[0])

        def weighted_median(arr):
            """
            Find weighted median of sorted array arr where each element is [coord, weight].
            Uses two-pointer elimination to find a coordinate where cumulative weight
            is balanced.
            """
            l, r = 0, len(arr) - 1
            while l < r:
                if arr[l][1] < arr[r][1]:
                    arr[r][1] -= arr[l][1]
                    l += 1
                elif arr[l][1] > arr[r][1]:
                    arr[l][1] -= arr[r][1]
                    r -= 1
                else:
                    # equal weights, eliminate both
                    l += 1
                    r -= 1
            return arr[l][0]

        # Compute medians in rotated space
        posx = weighted_median(A)
        posy = weighted_median(B)

        # Convert back to original coordinates (truncate towards zero)
        xx = int((posx + posy) / 2)
        yy = int((posx - posy) / 2)

        # Check the four nearest integer points
        candidates = [
            (xx, yy),
            (xx + 1, yy),
            (xx, yy + 1),
            (xx + 1, yy + 1)
        ]

        best_cost = None
        best_point = (xx, yy)

        for x, y in candidates:
            cost = 0
            for u, v, t in C:
                # Chebyshev distance * count
                cost += max(abs(x - u), abs(y - v)) * t
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_point = (x, y)

        # Output the optimal warehouse position
        x, y = best_point[0], best_point[1]

        self.parameter["reference_answer"] = "{} {}".format(x, y)
        self.parameter["gold_answer"] = sum(max(abs(x - Xi), abs(y - Yi)) * Ti for Xi, Yi, Ti in zip(X, Y, T))
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            X_Y_T = "\n".join("X[{}]={} Y[{}]={} T[{}]={}".format(i, Xi, i, Yi, i, Ti) for i, (Xi, Yi, Ti) in enumerate(zip(self.parameter["X"], self.parameter["Y"], self.parameter["T"]))),
        )


    def _process(self, answer : Optional[str]) -> Optional[Tuple] :
        if answer is not None :
            answer = answer.strip()
            try :
                x, y = map(int, answer.split())
                return x, y
            except :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            x, y = processed_result

            answer, gold = sum(max(abs(x - Xi), abs(y - Yi)) * Ti for Xi, Yi, Ti in zip(self.parameter["X"], self.parameter["Y"], self.parameter["T"])), self.parameter["gold_answer"]
            assert gold <= answer, "Gold answer should be less than or equal to the answer"

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if answer == 0 :
                    assert gold == 0, "If answer is 0, gold should also be 0"
                    return self.rewards["rewarding_weight"] * 1.0
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]