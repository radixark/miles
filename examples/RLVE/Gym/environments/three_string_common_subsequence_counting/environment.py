import random
import functools
from typing import Optional
from Gym.environment import VerifiableEnvironment


class ThreeStringCommonSubsequenceCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3856
    prompt_template = \
r"""There are three strings A, B, and C:
A: {A}  
B: {B}  
C: {C}

A string T is called a **subsequence** of another string S if T can be obtained from S by deleting zero or more characters without changing the order of the remaining characters. What is the number of **non-empty strings** that are subsequences of **A, B, and C simultaneously**?"""
    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the ThreeStringCommonSubsequenceCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 3, "MAX_N should be greater than or equal to 3"

        S = self.parameter["S"] = []
        a_probability = random.random()
        for _ in range(3) :
            length = random.randint(3, MAX_N)
            S.append("".join("a" if random.random() < a_probability else "b" for _ in range(length)))
        

        A, B, C = S[0], S[1], S[2]

        # Lengths
        n, m, k = len(A), len(B), len(C)

        # 1-based padding so we can use position 0 as “before start”
        A = '#' + A
        B = '#' + B
        C = '#' + C

        # Build next-occurrence tables of size (length+1)×2
        nextA = [[0]*2 for _ in range(n+1)]
        nextB = [[0]*2 for _ in range(m+1)]
        nextC = [[0]*2 for _ in range(k+1)]

        for u in range(n-1, -1, -1):
            # copy from the “next” row
            nextA[u] = nextA[u+1].copy()
            # record that char A[u+1] next appears at position u+1
            nextA[u][ord(A[u+1]) - ord('a')] = u+1

        for v in range(m-1, -1, -1):
            nextB[v] = nextB[v+1].copy()
            nextB[v][ord(B[v+1]) - ord('a')] = v+1

        for w in range(k-1, -1, -1):
            nextC[w] = nextC[w+1].copy()
            nextC[w][ord(C[w+1]) - ord('a')] = w+1

        # DFS with memoization: count all common substrings starting from positions (u,v,w)
        @functools.lru_cache(None)
        def dfs(u, v, w):
            total = 1  # count the “empty extension” here; we'll subtract it off at the end
            for ch in range(2):
                nu = nextA[u][ch]
                nv = nextB[v][ch]
                nw = nextC[w][ch]
                if nu and nv and nw:
                    total += dfs(nu, nv, nw)
            return total

        # Subtract 1 to exclude the empty substring
        self.parameter["reference_answer"] = dfs(0, 0, 0) - 1
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            A = self.parameter["S"][0],
            B = self.parameter["S"][1],
            C = self.parameter["S"][2],
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
                if self.parameter["reference_answer"] == 0 :
                    return self.rewards["rewarding_weight"] * int(processed_result == 0)
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]