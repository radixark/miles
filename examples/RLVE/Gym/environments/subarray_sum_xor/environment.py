import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SubarraySumXor_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3760
    prompt_template = \
r"""You are given an array A of {N} integers: {A}
This array has {N} × ({N} + 1) / 2 contiguous subarrays. For each subarray, compute its sum; then, output the **bitwise XOR** of all these subarray sums."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the SubarraySumXor_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        A = self.parameter["A"] = [random.randint(0, N) for _ in range(N)]


        # build prefix sums S[0..N]
        S = [0] * (N + 1)
        for i in range(1, N + 1):
            S[i] = S[i - 1] + A[i - 1]
        mx = S[N]

        # count how many times each prefix‐sum value appears (excluding S[0])
        cnt = [0] * (mx + 1)
        for i in range(1, N + 1):
            cnt[S[i]] += 1

        # scnt[v] = sum of cnt[0..v]
        scnt = [0] * (mx + 1)
        scnt[0] = cnt[0]
        for v in range(1, mx + 1):
            scnt[v] = scnt[v - 1] + cnt[v]

        ans = 0
        # for each bit j, count how many subarray‐sums have that bit = 1
        for j in range(mx.bit_length()):
            K = 1 << j
            M = 1 << (j + 1)

            # f[v] = number of earlier prefix‐sums s' with (v - s') in [K, M-1]
            f = [0] * (mx + 1)
            for v in range(mx + 1):
                # f[v - M] or 0 if out of range
                prev = f[v - M] if v >= M else 0
                # scnt[v - K] counts s' ≤ v-K
                add1 = scnt[v - K] if v >= K else 0
                # subtract those with s' ≤ v-M
                sub1 = scnt[v - M] if v >= M else 0
                f[v] = prev + add1 - sub1

            # g[v] = number of later prefix‐sums s' with (s' - v) in [K, M-1]
            g = [0] * (mx + 1)
            for v in range(mx, -1, -1):
                # g[v + M] or 0 if out of range
                prev = g[v + M] if v + M <= mx else 0
                # scnt[min(mx, v+M-1)] - scnt[min(mx, v+K-1)]
                hi = v + M - 1
                lo = v + K - 1
                add2 = scnt[hi] if hi <= mx else scnt[mx]
                sub2 = scnt[lo] if lo <= mx else scnt[mx]
                g[v] = prev + add2 - sub2

            # sum up f[S[i]] + g[S[i]] for i=1..N, then divide by 2 to get the # of subarrays
            res = 0
            for i in range(1, N + 1):
                sv = S[i]
                res += f[sv] + g[sv]
            res //= 2

            # if that count is odd, set bit j in ans
            if res & 1:
                ans |= K

        # finally, include the subarrays that start from index 1 (i.e. S[i] - S[0] = S[i])
        for i in range(1, N + 1):
            ans ^= S[i]

        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = ", ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"], start = 1)),
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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]