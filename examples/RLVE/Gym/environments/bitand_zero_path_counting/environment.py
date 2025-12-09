import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class BitAndZero_PathCounting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a **directed graph** with an **infinite number of vertices**, where each vertex is labeled with a non-negative integer: `0`, `1`, `2`, ...

There is a directed edge from vertex `s` to vertex `t` if and only if:
- `s < t`, and
- `s & t = 0` (where `&` denotes the bitwise AND operation)

Please compute the number of **distinct paths** from vertex `{S}` to vertex `{T}`. Give the result **modulo {MOD}**.
Note that the two vertices labels are provided in **binary (base-2)** representation.

**Output Format:** Your final answer should be a single integer — the number of distinct paths modulo `{MOD}`."""
    MOD = 10000

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the BitAndZero_PathCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    
    def _generate_helper(self) -> None :
        assert "max_length" in self.parameter, "max_length is required in parameter"
        max_length = self.parameter["max_length"]
        assert max_length >= 1, "max_length should be greater than or equal to 1"

        S = "1" + "".join(str(random.randint(0, 1)) for _ in range(random.randint(1, max_length) - 1))
        T = "1" + "".join(str(random.randint(0, 1)) for _ in range(random.randint(1, max_length) - 1))

        if len(S) > len(T) or (len(S) == len(T) and S > T) :
            S, T = T, S
            # Ensure S <= T
        self.parameter["S"], self.parameter["T"] = S, T


        MOD = self.MOD

        def Mult(a: int, b: int) -> int:
            return (a * b) % MOD

        def Add(a: int, b: int) -> int:
            s = a + b
            return s - MOD if s >= MOD else s

        S = list(map(int, S))
        T = list(map(int, T))
        N, M = len(S), len(T)

        if M > N:
            S = [0] * (M - N) + S
        else:
            assert M == N

        G = [[[0, 0] for _ in range(M)] for __ in range(2)]
        for st in (0, 1):
            G[st][0][st] = 1
            for i in range(1, M):
                G[st][i][0] = Add(G[st][i-1][0], G[st][i-1][1])
                G[st][i][1] = G[st][i-1][0]

        H = 1
        while H <= M and S[H-1] == 0:
            H += 1

        F = [[0] * M for _ in range(M + 1)]
        F[1][0] = 1

        for i in range(2, M + 1):
            for x in range(0, i - 1):
                bit = T[i-1]
                if i <= H:
                    F[i][x+1] = Add(F[i][x+1], Mult(F[i-1][x], G[1][x+1][bit]))
                if i < H:
                    total = Add(G[0][x][bit], G[1][x][bit])
                    F[i][x]   = Add(F[i][x],   Mult(F[i-1][x], total))
                if i > H:
                    F[i][x]   = Add(F[i][x],   Mult(F[i-1][x], G[S[i-1]][x][bit]))

        ans = 0
        for x in range(0, M):
            ans = Add(ans, F[M][x])
        self.parameter["reference_answer"] = ans
    

    def _generate(self) -> None :
        while True :
            self._generate_helper()
            if self.parameter["reference_answer"] not in (0, 1) :
                break
    
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            S = self.parameter["S"],
            T = self.parameter["T"],
            MOD = self.MOD,
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
            if not (0 <= processed_result < self.MOD) :
                return self.rewards["wrong_range"]
            
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]