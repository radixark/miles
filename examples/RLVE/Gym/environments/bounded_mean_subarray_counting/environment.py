import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class BoundedMeanSubarrayCounting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Given an array A of length {N}:
{A}

How many nonempty contiguous subarrays have a mean greater than or equal to {K}?

**Output Format:** Your final answer should be a single integer — the total number of nonempty subarrays of A whose mean is greater than or equal to {K}."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the BoundedIntervalCounting_Environment instance.
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
        assert N >= 2, "N should be greater than or equal to 2"

        A = self.parameter["A"] = [random.randint(0, N) for _ in range(N)]
        K = self.parameter["K"] = random.randint(min(A), max(A))


        v = [0] * (N + 1)
        for i in range(1, N + 1) :
            v[i] = v[i - 1] + A[i - 1] - K

        tmp = [0] * (N + 1)

        res = 0
        def cdq(l, r) :
            nonlocal res
            if l >= r :
                return
            mid = (l + r) // 2
            cdq(l, mid)
            cdq(mid + 1, r)

            i, j = l, mid + 1
            sum_left = 0
            for k in range(l, r + 1) :
                if j > r or (i <= mid and v[i] <= v[j]) :
                    sum_left += 1
                    tmp[k] = v[i]
                    i += 1
                else :
                    res += sum_left
                    tmp[k] = v[j]
                    j += 1

            for k in range(l, r + 1) :
                v[k] = tmp[k]

        cdq(0, N)
        assert res > 0
        self.parameter["reference_answer"] = res
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = " ".join(map(str, self.parameter["A"])),
            K = self.parameter["K"],
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
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]