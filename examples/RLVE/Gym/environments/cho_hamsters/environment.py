import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class ChoHamsters_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3502
    prompt_template = \
r"""You are given {N} strings, listed below (it is guaranteed that for all i ≠ j, the string S[i] is **NOT** a contiguous substring of S[j]):
{strings}

Please construct a string T such that the **sum** (for all i) of `counting(T, S[i])` is **at least {M}**, where `counting(T, s)` is the number of (possibly overlapping) occurrences of the string `s` in `T`.
Try your best to **minimize the length** of such a string `T`. Output a single integer — the minimum possible length of `T`."""

    def __init__(self,
                 length_multiple_min : int = 2, length_multiple_max : int = 3,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the ChoHamsters_Environment instance.
        """
        super().__init__(**kwargs)

        self.length_multiple_min, self.length_multiple_max = length_multiple_min, length_multiple_max
        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 1, "N should be greater than or equal to 1"
        
        while True :
            S = self.parameter["S"] = []
            for _ in range(N) :
                length = random.randint(N * self.length_multiple_min, N * self.length_multiple_max)
                a_probability = random.random()
                Si = "".join("a" if random.random() < a_probability else "b" for _ in range(length))
                S.append(Si)
            if all(Si not in Sj for i, Si in enumerate(S) for j, Sj in enumerate(S) if i != j) :
                break
        
        assert "MAX_M" in self.parameter, "MAX_M is required in parameter"
        MAX_M = self.parameter["MAX_M"]
        assert MAX_M >= 1, "MAX_M should be greater than or equal to 1"
        
        M = self.parameter["M"] = random.randint(1, MAX_M)


        # Compute prefix-function (KMP) for each string in S
        # pi[i][k] = length of longest proper prefix of S[i] which is also a suffix of S[i][:k+1]
        pi = []
        for s in S:
            L = len(s)
            p = [0] * L
            j = 0
            for i in range(1, L):
                while j > 0 and s[j] != s[i]:
                    j = p[j-1]
                if s[j] == s[i]:
                    j += 1
                p[i] = j
            pi.append(p)

        # Determine an upper bound INF based on maximum possible cost:
        # worst case, no overlaps, each added name costs its full length,
        # so M * max_len + something.
        max_len = max(len(s) for s in S)
        INF = M * max_len + 1

        # Build the transition matrix Tra of size (N+1) x (N+1)
        # Node 0 is the start; nodes 1..N correspond to S[0]..S[N-1]
        Tra = [[INF] * (N+1) for _ in range(N+1)]
        
        # From start (0) to each name x: cost = full length of name x
        for x in range(1, N+1):
            Tra[0][x] = len(S[x-1])
        # From any name x back to start is impossible (set to INF)
        # Tra[x][0] already INF
        
        # Precompute transition costs between names
        # Tra[x][y] = extra letters needed to append name y after name x
        for x in range(1, N+1):
            sx = S[x-1]
            len_x = len(sx)
            for y in range(1, N+1):
                sy = S[y-1]
                len_y = len(sy)
                # Find overlap: longest suffix of sx matching prefix of sy
                j = 0
                # iterate over sx[1..end] (0-based: positions 1..len_x-1)
                for i in range(1, len_x):
                    while j > 0 and sy[j] != sx[i]:
                        j = pi[y-1][j-1]
                    if sy[j] == sx[i]:
                        j += 1
                # j is the overlap length
                Tra[x][y] = len_y - j

        # Matrix multiplication in min-plus (tropical) semiring
        def mat_mult(A, B):
            C = [[INF] * (N+1) for _ in range(N+1)]
            for i in range(N+1):
                for j in range(N+1):
                    # we can skip if A[i][j] is INF
                    aij = A[i][j]
                    if aij == INF:
                        continue
                    row_i = C[i]
                    bj = B[j]
                    for k in range(N+1):
                        v = aij + bj[k]
                        if v < row_i[k]:
                            row_i[k] = v
            return C

        # Fast exponentiation: compute Ans = Tra^M
        # Ans initially Tra^1
        Ans = [row[:] for row in Tra]
        exp = M - 1  # we already account for one application of Tra
        base = [row[:] for row in Tra]
        while exp > 0:
            if exp & 1:
                Ans = mat_mult(Ans, base)
            base = mat_mult(base, base)
            exp >>= 1

        # The answer is the minimum cost from start (0) to any name after M transitions
        result = min(Ans[0][1:])  # ignore Ans[0][0]
        self.parameter["reference_answer"] = result
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            strings = "\n".join("S[{}]={}".format(i, Si) for i, Si in enumerate(self.parameter["S"])),
            M = self.parameter["M"],
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