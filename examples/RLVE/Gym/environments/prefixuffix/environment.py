import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Prefixuffix_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3546
    prompt_template = \
r"""Define two strings S1 and S2 to be **equivalent** if one can be obtained from the other by moving a suffix to the front (i.e., performing a cyclic shift). For example, the strings "ababba" and "abbaab" are equivalent because "ababba" = "ab" + "abba" and "abbaab" = "abba" + "ab"

You are given a string S of length {N}: {S}
Please output the largest integer L such that 2 × L ≤ {N}, and the L-prefix (i.e., the first L characters of S) and the L-suffix (i.e., the last L characters of S) are equivalent (see the definition above)."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the Prefixuffix_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "correct_answer": correct_answer,
            "wrong_answer": wrong_answer
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 4, "N should be greater than or equal to 4"
        
        a_probability = random.random()
        def generate_string(length : int) -> str :
            return "".join("a" if random.random() < a_probability else "b" for _ in range(length))

        L = random.randint(1, N // 2)
        L1 = random.randint(0, L)
        L2 = L - L1
        S1, S2 = generate_string(L1), generate_string(L2)
        self.parameter["S"] = S = (S1 + S2) + generate_string(N - 2 * L) + (S2 + S1)


        # Build interleaved string t[1..N], with t[0] a sentinel
        t = ['#'] * (N + 1)
        # fill odd positions with S[0], S[1], ...
        j = 1
        for i in range(N):
            if j <= N:
                t[j] = S[i]
            j += 2
        # fill even positions with S[N-1], S[N-2], ...
        j = 2
        for i in range(N - 1, -1, -1):
            if j <= N:
                t[j] = S[i]
            j += 2

        # p[i]: radius of the even-length palindrome centered between t[i] and t[i+1]
        p = [0] * (N + 1)
        # vis[k] = 1 iff there is a palindrome of radius exactly i at center i such that it touches t[0]
        vis = [0] * (N + 2)

        mr = 0       # rightmost reach of any palindrome seen so far
        mid2 = 0     # twice the center index of that palindrome

        # Manacher's algorithm for even-length palindromes on t
        for i in range(1, N):
            # mirror optimization
            if mid2 - i - 1 > 0 and mr - i - 1 > 0:
                p[i] = min(p[mid2 - i - 1], mr - i - 1)
            else:
                p[i] = 0
            # expand around center between i and i+1
            while i - p[i] >= 0 and i + 1 + p[i] <= N and t[i - p[i]] == t[i + 1 + p[i]]:
                p[i] += 1
            # update rightmost palindrome
            if i + 1 + p[i] > mr:
                mr = i + 1 + p[i]
                mid2 = 2 * i + 1
            # if it reaches the sentinel at t[0], mark vis
            if i == p[i]:
                vis[i + p[i]] = 1

        # Union-find to compute, for each starting point j, the max center i covering it
        f = list(range(N + 2))
        res = [0] * (N + 2)
        def find(x):
            while f[x] != x:
                f[x] = f[f[x]]
                x = f[x]
            return x

        # Populate res[j] = max i such that [j..i] is inside some palindrome
        for i in range(N - 1, 0, -1):
            start = i - p[i] + 1
            j = find(start)
            while j <= i:
                res[j] = i
                f[j] = find(j + 1)
                j = f[j]

        # Compute answer as the largest L ≤ N//2 where prefix and suffix are cyclically equivalent
        ans = 0
        # Case 1: using two-part palindromes
        for i in range(1, N + 1):
            if vis[i] and res[i + 1] != 0:
                # solve (2*res + 1 - (i+1)) / 2
                val = (2 * res[i + 1] + 1 - (i + 1)) // 2
                if val > ans:
                    ans = val
        # Case 2: trivial rotations within the first part
        for i in range(1, N + 1):
            if vis[i]:
                val = i // 2
                if val > ans:
                    ans = val

        assert L <= ans <= N // 2, "Computed answer is not within the expected range"
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], S = self.parameter["S"])


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