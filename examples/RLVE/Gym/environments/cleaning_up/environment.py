import math
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class CleaningUp_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2943
    prompt_template = \
r"""You are given {N} numbers A[1], A[2], ..., A[{N}]. The values are: {A}
You may divide these numbers (in order) into **consecutive non-empty batches**. Let the total number of batches be k, and let end[1], end[2], ..., end[k] (1 ≤ end[1] < end[2] < ... < end[k] = {N}) denote the last index of each batch. This means:
- Batch 1 contains A[1] to A[end[1]]
- Batch 2 contains A[end[1] + 1] to A[end[2]]
- ...
- Batch k contains A[end[k − 1] + 1] to A[end[k]] (with end[k] = {N})

Define the cost of a division as follows:
- For each batch i (1 <= i <= k), let K[i] be the number of **distinct** values in that batch.
- The total cost is the sum of K[i]^2 (i.e., the square of K[i]) over all batches.

Can you find a division that **minimizes the total cost**?

**Output Format:**
Output a single line: `end[1] end[2] ... end[k]` (space-separated, with `end[k] = {N}`).
Example: `1 2 {N}` means there are 3 batches ending at indices 1, 2, and {N}, respectively."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_beta : float = 5.0, rewarding_weight : float = 1.0,
                 **kwargs) :
        """
        Initialize the CleaningUp_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 4, "N should be greater than or equal to 4"

        while True :
            endpoints = random.sample(range(1, N), k = random.randint(1, N - 1))
            endpoints.sort()
            endpoints += [N]

            for i in range(len(endpoints) - 1, 0, -1) :
                endpoints[i] -= endpoints[i - 1]
            
            A = self.parameter["A"] = []
            for x in endpoints :
                assert x > 0
                number_range = 1
                while (number_range + 1) * (number_range + 1) <= x :
                    number_range += 1
                number_range = random.sample(range(1, N + 1), k = number_range)
                A.extend([random.choice(number_range) for _ in range(x)])
            assert len(A) == N
            

            # Read preferences P (1-indexed); set P[0]=0 as a harmless sentinel
            P = [0] * (N + 1)
            for i in range(1, N + 1):
                P[i] = A[i - 1]

            k = int(math.isqrt(N))  # sqrt(N)
            # Move-to-front list of last occurrences for up to k+1 distinct foods
            last = [-1] * (k + 2)   # +2 to be safe for j=k+1 during shifting
            last[0] = 0

            # DP: f[i] = minimal total cost for first i cows
            f = [None] * (N + 1)
            f[0] = 0

            for i in range(1, N + 1):
                x = P[i]

                # Find position j in move-to-front list for current type (or insertion point)
                j = 0
                while j <= k and last[j] != -1 and P[last[j]] != x:
                    j += 1

                # Move-to-front: shift [0..j-1] right by one, put i at front
                while j > 0:
                    last[j] = last[j - 1]
                    j -= 1
                last[0] = i

                # Transition: consider segments ending at i with up to k distinct foods
                best = None
                j = 1
                while j <= k and last[j] != -1:
                    prev = f[last[j]]
                    cand = None if prev is None else prev + j * j
                    if best is None or (cand is not None and cand < best):
                        best = cand
                    j += 1

                f[i] = best

            self.parameter["gold_answer"] = f[N]
            assert self.parameter["gold_answer"] > 0

            if self.parameter["gold_answer"] < min(N, len(set(A)) ** 2) :
                break
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = ", ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"], start = 1)),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                if not answer_array :
                    return None
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            N = self.parameter["N"]

            ends = processed_result
            if not (1 <= len(ends) <= N) :
                return self.rewards["invalid_solution"]
            for i in range(len(ends)) :
                if not (1 <= ends[i] <= N) :
                    return self.rewards["invalid_solution"]
                if i and not (ends[i - 1] < ends[i]) :
                    return self.rewards["invalid_solution"]
            if ends[-1] != N :
                return self.rewards["invalid_solution"]
            
            A = [None] + self.parameter["A"]
            answer = 0
            last = 0
            for end in ends :
                K = len(set(A[last + 1 : end + 1]))
                answer += K ** 2
                last = end
            gold = self.parameter["gold_answer"]
            assert 0 < gold <= answer, "Gold answer should be less than or equal to the computed answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]