import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Minimum_DominatingInterval_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""There are {N} points labeled 1 through {N} on a line. You are given {M} intervals [L[i], R[i]] (1 <= L[i] <= R[i] <= {N}), each with a cost C[i]:
{intervals}

Please select {K} distinct points such that each selected point is **covered by at least one** of the intervals.
The cost of a selection is the sum of the costs (C[i]) of all intervals that cover at least one of the selected points.
Try your best to minimize the total cost of the selection.

**Output Format:** Your final answer should be a single line containing the {K} selected points, separated by spaces. Example: {first_K_points} (do **NOT** include quotes or backticks)."""

    def __init__(self,
                 cost_range : int = 10,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Minimum_DominatingInterval_Environment instance.
        """
        super().__init__(**kwargs)

        self.cost_range = cost_range
        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert M >= 2, "M should be greater than or equal to 2"

        all_intervals = [(l, r, random.randint(1, self.cost_range)) for l in range(1, N + 1) for r in range(l, N + 1)]
        assert len(all_intervals) == (N * (N + 1) // 2)
        intervals = self.parameter["intervals"] = random.sample(all_intervals, min(len(all_intervals), M))

        assert "K_density" in self.parameter, "K_density is required in parameter"
        K_density = self.parameter["K_density"]
        assert 0.0 <= K_density <= 1.0, "K_density should be between 0.0 and 1.0"
        def full_point_set_size() -> int :
            dominated = set()
            for interval in self.parameter["intervals"] :
                Li, Ri = interval[0], interval[1]
                dominated.update(range(Li, Ri + 1))
            return len(dominated)
        K = self.parameter["K"] = max(1, int(K_density * full_point_set_size()))


        L, R, C = zip(*intervals)

        Sum_Ci = [[0] * (N + 1) for l in range(N + 1)]
        for i in range(M) :
            Li, Ri, Ci = L[i], R[i], C[i]
            Sum_Ci[Li][Ri] = Sum_Ci[Li][Ri] + Ci
        for l in range(1, N + 1) :
            for r in range(N - 1, 0, -1) :
                Sum_Ci[l][r] += Sum_Ci[l][r + 1]

        dpF = [[None] * (N + 1) for k in range(0, K + 1)]
        dpG = [[None] * (N + 1) for k in range(0, K + 1)]
        for i in range(1, N + 1) :
            if not any (Li <= i and i <= Ri for Li, Ri in zip(L, R)) :
                continue
            dpF[1][i] = 0
            for l in range(1, i + 1) :
                dpF[1][i] += Sum_Ci[l][i]
        for k in range(2, K + 1) :
            for i in range(1, N + 1) :
                if not any (Li <= i and i <= Ri for Li, Ri in zip(L, R)) :
                    continue
                Sum = 0
                for j in range(i, 0, -1) :
                    Sum += Sum_Ci[j][i]
                    if dpF[k - 1][j - 1] is not None :
                        val = dpF[k - 1][j - 1] + Sum
                        if dpF[k][i] is None or val < dpF[k][i] :
                            dpF[k][i] = val
                            dpG[k][i] = j - 1

        last = None
        for i in range(1, N + 1) :
            if dpF[K][i] is None :
                continue
            if dpF[K][i] is not None and (last is None or dpF[K][i] < dpF[K][last]) :
                last = i
        pickeds = []
        for k in range(K, 0, -1) :
            assert last is not None
            pickeds.append(last)
            last = dpG[k][last]
        assert last is None
        pickeds.reverse()

        self.parameter["reference_answer"] = " ".join(map(str, pickeds))
        self.parameter["gold_answer"] = sum(C[i] for i in range(M) if any(L[i] <= picked and picked <= R[i] for picked in pickeds))
        assert self.parameter["gold_answer"] > 0
    
    def _prompt_generate(self) -> str :
        L, R, C = zip(*self.parameter["intervals"])
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            K = self.parameter["K"],
            intervals = "\n".join("L[{}]={}, R[{}]={}, C[{}]={}".format(i, L[i], i, R[i], i, C[i]) for i in range(self.parameter["M"])),
            first_K_points = " ".join(map(str, range(1, self.parameter["K"] + 1))),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List] :
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
            assert isinstance(processed_result, list), "processed_result should be a list"

            pickeds = processed_result
            if len(pickeds) != self.parameter["K"] :
                return self.rewards["invalid_solution"]
            if len(set(pickeds)) != self.parameter["K"] :
                return self.rewards["invalid_solution"]
        
            L, R, C = zip(*self.parameter["intervals"])
            if not all(any(Li <= picked <= Ri for Li, Ri in zip(L, R)) for picked in pickeds) :
                return self.rewards["invalid_solution"]
            
            gold = self.parameter["gold_answer"]
            answer = sum(C[i] for i in range(self.parameter["M"]) if any(L[i] <= picked and picked <= R[i] for picked in pickeds))
            assert gold <= answer, "answer should be greater than or equal to gold"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]