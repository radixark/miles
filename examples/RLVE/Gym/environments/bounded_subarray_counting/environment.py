import random
import bisect
from typing import Optional
from Gym.environment import VerifiableEnvironment


class BoundedSubarrayCounting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Given an array A of length {N}:
{A}

Repeat array A {M} times to form a new array B of length {N} * {M} = {NM}. In the new array B, how many (nonempty) contiguous subarrays have a total sum less than or equal to {K}?

**Output Format:** Your final answer should be a single integer — the total number of (nonempty) subarrays in B whose sum is less than or equal to {K}."""

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
        assert N >= 1, "N should be greater than or equal to 1"

        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert M >= 2, "M should be greater than or equal to 2"

        A = self.parameter["A"] = [random.randint(1, N) for _ in range(N)]
        K = self.parameter["K"] = random.randint(max(A), sum(A) * M)


        # build prefix sums s[0..n]
        s = [0] * (N + 1)
        for i in range(1, N + 1):
            s[i] = s[i - 1] + A[i - 1]
        total = s[N]

        ans = 0
        # precompute m*(m-1)/2 * n for the “full‐span” case
        mmn = M * (M - 1) // 2 * N

        for i in range(1, N + 1) :
            si = s[i]
            if si < K :
                # how many *full* repeats we can append after position i without exceeding k
                d = (K - si) // total
                if d < M - 1 :
                    # contributions from using 0,1,...,d full copies
                    ans += i * (d + 1) + d * (d + 1) // 2 * N

                    # partial in the (d+1)-th copy
                    e = (K - si) % total
                    # find smallest j with s[j] >= total - e
                    j = bisect.bisect_left(s, total - e)
                    # for each of the remaining (m-1-d) copies, we can take up to (n-j) more elements
                    ans += (i + d * N + (N - j)) * (M - 1 - d)
                else :
                    # we can take all m copies plus all possible “full-span” subarrays
                    ans += i * M + mmn
            else :
                # even the prefix [1..i] exceeds k, so only shorter endings count
                # find j so that s[i] - s[j] <= k  =>  s[j] >= s[i] - k
                j = bisect.bisect_left(s, si - K)
                ans += (i - j) * M

        self.parameter["reference_answer"] = ans
        assert ans > 0
    
    def _prompt_generate(self) -> str :
        N, M = self.parameter["N"], self.parameter["M"]
        return self.prompt_template.format(
            N = N,
            M = M,
            NM = N * M,
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