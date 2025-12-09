import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class LIZ_Lollipop_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3514
    prompt_template = \
r"""You are given an array `A` of length {N}: {A}
Each element in `A` is either 1 or 2, and the total sum of the array is {S}.

You need to output {S} lines. For the i-th line (1 ≤ i ≤ {S}), output two integers `l` and `r` (0-indexed, inclusive), separated by a space:
- If there exists a contiguous subarray A[l : r + 1] (Python-style slicing, so it includes A[l] & A[r] but NOT A[r + 1]) such that the sum of its elements is exactly `i`, output `l` and `r`.
- If no such subarray exists, output `-1 -1`."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(correct/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the LIZ_Lollipop_Environment instance.
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
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        two_probability = random.random()
        A = self.parameter["A"] = [2 if random.random() < two_probability else 1 for _ in range(N)]
        

        A = [0] + A.copy() # Convert to 1-indexed
        # prefix sums
        pref = [0] * (N + 1)
        for i in range(1, N + 1):
            pref[i] = pref[i-1] + A[i]
        S = pref[N]

        # for each sum k (0..S), store one interval [l[k],r[k]] that sums to k, if known
        l = [0] * (S + 3)
        r = [0] * (S + 3)
        # Max[0] = max even sum seen, Max[1] = max odd sum seen
        Max = [-1, -1]

        def up(val, ll, rr):
            p = val & 1
            if val > Max[p]:
                Max[p] = val
                l[val] = ll
                r[val] = rr

        # record all prefixes and suffixes
        for i in range(1, N):
            up(S - pref[i], i+1, N)   # suffix sum
            up(pref[i], 1, i)         # prefix sum
        # whole string
        up(S, 1, N)

        # propagate downward from S to 1 by deleting a 1 or 2 from one end
        for k in range(S, 0, -1):
            if l[k] == 0 and r[k] == 0:
                pl, pr = l[k+2], r[k+2]
                if pl and pr:
                    ll, rr = pl, pr
                    if A[pl] == 2:
                        ll += 1
                    elif A[pr] == 2:
                        rr -= 1
                    else:
                        ll += 1
                        rr -= 1
                    l[k], r[k] = ll, rr

        self.parameter["reference_answer"] = []
        self.parameter["existence"] = []
        for x in range(1, S + 1) :
            # impossible if x > total sum, or we never saw any substring of that parity ≥ x
            if x > S or x > Max[x & 1]:
                self.parameter["reference_answer"].append("-1 -1")
                self.parameter["existence"].append(False)
            else:
                self.parameter["reference_answer"].append("{} {}".format(l[x] - 1, r[x] - 1))
                self.parameter["existence"].append(True)
        self.parameter["reference_answer"] = "\n".join(self.parameter["reference_answer"])
    

    def _prompt_generate(self) -> str :
        A = self.parameter["A"]
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(A)),
            S = sum(A),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answers = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        l, r = map(int, line.split())
                        answers.append((l, r))
                return answers
            except :
                return None
        else :
            return None


    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            S = [0] * self.parameter["N"]
            for i, Ai in enumerate(self.parameter["A"]) :
                S[i] = (S[i - 1] if i - 1 >= 0 else 0) + Ai
            
            if len(processed_result) != S[self.parameter["N"] - 1] :
                return self.rewards["invalid_solution"]
            assert len(processed_result) == len(self.parameter["existence"]), "Length of processed result does not match existence list"

            correct = 0
            for x in range(1, len(processed_result) + 1) :
                l, r = processed_result[x - 1]
                existence = self.parameter["existence"][x - 1]
                if not ((l, r) == (-1, -1) or (0 <= l <= r < self.parameter["N"])) :
                    return self.rewards["invalid_solution"]
                if existence :
                    correct += int((0 <= l <= r < self.parameter["N"]) and (S[r] - (S[l - 1] if l > 0 else 0) == x))
                else :
                    if 0 <= l <= r < self.parameter["N"] :
                        assert S[r] - (S[l - 1] if l > 0 else 0) != x
                    correct += int((l, r) == (-1, -1))
            
            if self.rewards["rewarding_strategy"] == "(correct/all)^beta" :
                return self.rewards["rewarding_weight"] * ((correct / len(processed_result)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "correct=all" :
                return self.rewards["rewarding_weight"] * (correct == len(processed_result))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]