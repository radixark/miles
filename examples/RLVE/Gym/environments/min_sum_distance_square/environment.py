import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MinSumDistanceSquare_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3162
    prompt_template = \
r"""There are {N} groups of points located on the x-axis. The coordinates of each group are given as follows:
{points}

Your task is to choose a point X on the x-axis. For each group i (0 ≤ i < {N}), define cost[i] as the square of the minimum distance from X to any point in that group: cost[i] = (min(abs(X - x_i[j])))^2, where x_i[j] is the j-th point in group i.
Please find the value of X that minimizes the total cost, i.e., the sum of all cost[i].

It can be shown that there exists an optimal solution X = X' / {N}, where X' is an integer. Please output this integer X'."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinSumDistanceSquare_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def compute_toal_cost(self, X_prime : int) -> int :
        # (X_prime / N - x)^2 = (X_prime - N * x)^2 / N^2
        return sum(min((X_prime - self.parameter["N"] * x) ** 2 for x in xs) for xs in self.parameter["points"])
    

    def _generate(self) -> None :
        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert M >= 2, "M should be greater than or equal to 2"
        
        coordinatenates = [random.randint(-M, +M) for _ in range(M)]

        N = self.parameter["N"] = random.randint(2, M)
        belongings = list(range(N)) + [random.randint(0, N - 1) for _ in range(M - N)]
        random.shuffle(belongings)
        
        self.parameter["points"] = points = [[] for _ in range(N)]
        for coordinate, belonging in zip(coordinatenates, belongings) :
            points[belonging].append(coordinate)
        

        F = [[] for _ in range(N)]                # F[i] = coordinates producing part i (0-indexed)

        for p, xs in enumerate(points) :
            assert len(xs) > 0, "Each group must have at least one point"
            for x in xs :
                F[p].append(x)
            F[p].sort()  # sort each group

        events = []                               # consecutive-pair events
        O = 0                                     # Σ X_i^2
        E = 0                                     # Σ X_i

        for lst in F:
            lst.sort()
            O += lst[0] * lst[0]
            E += lst[0]
            for j in range(1, len(lst)):
                events.append((lst[j - 1], lst[j]))

        # sort by midpoint   (a+b)
        events.sort(key=lambda ab: ab[0] + ab[1])

        best_value = N * O - E * E                # current minimal n*O - E^2
        best_E = E

        for a, b in events:
            O += b * b - a * a
            E += b - a
            value = N * O - E * E
            if value < best_value:
                best_value = value
                best_E = E

        self.parameter["reference_answer"] = best_E
        self.parameter["gold_answer"] = self.compute_toal_cost(best_E)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            points = "\n".join("Group {}: {}".format(i, " ".join(map(str, xs))) for i, xs in enumerate(self.parameter["points"])),
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
            gold, answer = self.parameter["gold_answer"], self.compute_toal_cost(processed_result)
            assert 0 <= gold <= answer, "gold_answer should be less than or equal to answer"
            
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