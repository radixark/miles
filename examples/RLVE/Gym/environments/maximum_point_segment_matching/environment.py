import random
import bisect
from typing import Optional, List, Tuple
from Gym.environment import VerifiableEnvironment


class MaximumPointSegmentMatching_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given {C} points, indexed from 0 to {C_minus_1}:
{points}

You are also given {N} segments (each represented as a closed interval [l, r], meaning both endpoints are inclusive), indexed from 0 to {N_minus_1}:
{segments}

A valid matching is a set of pairs (c, n), where:
- `c` is the index of a point (0 <= c < {C}) and `n` is the index of a segment (0 <= n < {N}),
- point `c` lies within segment `n` (i.e., the point is contained in the segment),
- **no point is matched to more than one segment**, and **no segment is matched to more than one point**.

I want you to find the **maximum matching** between points and segments.
The number of your output lines should equal the size of your matching. Output one line for each matched pair - each line should contain two integers `c` and `n`, separated by a space, indicating a matched pair (point index, segment index)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MaximumPointSegmentMatching_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_C_N" in self.parameter, "MAX_C_N is required in parameter"
        MAX_C_N = self.parameter["MAX_C_N"]
        assert MAX_C_N >= 1, "MAX_C_N should be greater than or equal to 1"

        C = self.parameter["C"] = random.randint(2, MAX_C_N)
        N = self.parameter["N"] = random.randint(2, MAX_C_N)
        while True :
            points = self.parameter["points"] = [random.randint(0, MAX_C_N) for _ in range(C)]
            
            segments = self.parameter["segments"] = []
            for _ in range(N) :
                length = random.randint(0, MAX_C_N)
                l = random.randint(0, MAX_C_N - length)
                r = l + length
                segments.append((l, r))
            

            # Read the times T_i when each chicken is available
            times = points.copy()
            
            # Read the intervals [A_j, B_j] during which each cow can cross
            intervals = segments.copy()
            
            # Sort chicken times for binary search
            times.sort()
            
            # Sort cows by their end time ascending; if tied, by start time descending
            intervals.sort(key=lambda interval: (interval[1], -interval[0]))
            
            ans = 0
            # Greedily assign each cow the earliest available chicken in its interval
            for A, B in intervals:
                # Find the first chicken time >= A
                idx = bisect.bisect_left(times, A)
                # If that chicken is also <= B, match them
                if idx < len(times) and times[idx] <= B:
                    ans += 1
                    # Remove that chicken from availability
                    times.pop(idx)
            
            if ans > 0 :
                self.parameter["gold_answer"] = ans
                break
    

    def _prompt_generate(self) -> str :
        C = self.parameter["C"]
        N = self.parameter["N"]
        return self.prompt_template.format(
            C = C,
            C_minus_1 = C - 1,
            points = "\n".join("point {}: {}".format(i, p) for i, p in enumerate(self.parameter["points"])),
            N = N,
            N_minus_1 = N - 1,
            segments = "\n".join("segment {}: [{}, {}]".format(i, l, r) for i, (l, r) in enumerate(self.parameter["segments"])),
        )


    def _process(self, answer : Optional[str]) -> Optional[List[Tuple[int, int]]] :
        if answer is not None :
            answer = answer.strip()
            try :
                operations = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        c, n = map(int, line.split())
                        operations.append((c, n))
                return operations
            except :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            used_points, used_segments = [False] * self.parameter["C"], [False] * self.parameter["N"]
            for c, n in processed_result :
                if not (0 <= c < self.parameter["C"]) or not (0 <= n < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                if not (self.parameter["segments"][n][0] <= self.parameter["points"][c] <= self.parameter["segments"][n][1]) :
                    return self.rewards["invalid_solution"]
                if used_points[c] or used_segments[n] :
                    return self.rewards["invalid_solution"]
                used_points[c] = used_segments[n] = True
            answer, gold = len(processed_result), self.parameter["gold_answer"]
            assert 0 <= answer <= gold, "Answer should be between 0 and gold_answer"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]