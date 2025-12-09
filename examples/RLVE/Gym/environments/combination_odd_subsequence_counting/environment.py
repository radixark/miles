import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class CombinationOddSubsequenceCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3773
    prompt_template = \
r"""You are given a sequence of **distinct** integers: {array}

Please count the number of subsequences (not necessarily contiguous, but the order must be preserved) a[1], ..., a[k] such that:
1. k ≥ 2 (the subsequence must have at least two elements);
2. C(a[1], a[2]) × C(a[2], a[3]) × ... × C(a[k−1], a[k]) is **odd**, where C(x, y) denotes the binomial coefficient "x choose y".

**Output Format:** A single integer — the number of such valid subsequences."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the CombinationOddSubsequenceCounting_Environment instance.
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

        A = self.parameter["A"] = random.sample(range(1, 2 * N), N)
        random.shuffle(A)


        max_val = max(A)
        T = [-1] * (max_val + 1)
        for i, v in enumerate(A):
            T[v] = i

        # f[i] = number of non-increasing subsequences starting at i (including length-1)
        f = [0] * N
        ans = 0

        # DP from right to left
        for i in range(N - 1, -1, -1):
            mask = A[i]
            cnt = 1  # the subsequence [A[i]] itself
            # enumerate all non-zero proper submasks j of mask
            j = mask & (mask - 1)
            while j:
                idx = T[j]
                # if that value appears later in the sequence, extend subsequences
                if idx > i:
                    cnt += f[idx]
                j = mask & (j - 1)
            f[i] = cnt
            ans += cnt

        # subtract the length-1 subsequences to count only those of length >= 2
        ans -= N
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(array = " ".join(map(str, self.parameter["A"])))
    

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
                a, b = self.parameter["reference_answer"], processed_result
                if b == 0 :
                    return self.rewards["rewarding_weight"] * (a == 0)
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]