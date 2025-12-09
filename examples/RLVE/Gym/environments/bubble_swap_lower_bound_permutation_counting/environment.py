import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class BubbleSwapLowerBound_PermutationCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4769
    prompt_template = \
r"""Consider bubble sort on a permutation p[1..{N}] using the standard double loop:
```
for i = 1 to N:
  for j = 1 to N-1:
    if p[j] > p[j+1]: swap p[j], p[j+1]
```
It is known that the number of swaps performed by this algorithm is at least LB(p) = (abs(1 - p[1]) + abs(2 - p[2]) + ... + abs(N - p[N])) / 2. Tell me the number of permutations p of 1, 2, ..., {N} that satisfy BOTH:
1) The bubble sort swap count equals the lower bound: swaps(p) = LB(p).
2) p is lexicographically strictly greater than the given permutation P: {P}"""
    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the BubbleSwapLowerBound_PermutationCounting_Environment instance.
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
        assert N >= 3, "N should be greater than or equal to 3"

        self.parameter["P"] = P = list(range(1, N + 1))
        random.shuffle(P)


        q = P.copy()

        # Build Pascal triangle up to 2*N (inclusive), no modulo during building
        max_row = 2 * N
        C = []
        for i in range(max_row + 1):
            row = [0] * (i + 1)
            row[0] = 1
            row[-1] = 1
            for j in range(1, i):
                row[j] = C[i - 1][j - 1] + C[i - 1][j]
            C.append(row)

        def comb(n, m):
            if n < 0 or m < 0 or m > n:
                return 0
            # n <= max_row should always hold given how F is used
            return C[n][m]

        def F(i, j):
            # i, j are 0/1-based consistent with the original usage:
            # F(i-1, max(mx, v) + 1) in the loop with i from 1..N
            if not (i <= j <= N):
                return 0
            x = 2 * N - i - j - 1
            a = N - i - 1
            b = N - j - 2
            return comb(x, a) - comb(x, b)

        vis = [False] * (N + 2)  # 1..N used; N+1 safe guard
        ans = 0
        mx = 0
        mn = 1
        flag = False

        for i in range(1, N + 1):
            v = q[i - 1]
            if flag:
                continue
            ans += F(i - 1, max(mx, v) + 1)
            if mx > v and v > mn:
                flag = True
            mx = max(mx, v)
            vis[v] = True
            while mn <= N and vis[mn]:
                mn += 1
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            P = ", ".join("P[{}]={}".format(i, Pi) for i, Pi in enumerate(self.parameter["P"], start = 1)),
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
                if processed_result == 0 :
                    return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == 0)
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]