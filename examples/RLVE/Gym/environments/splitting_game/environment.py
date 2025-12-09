import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SplittingGame_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3185
    prompt_template = \
r"""There are {N} bottles of beans, indexed from 0 to {N_minus_1}. Initially, the i-th bottle contains P[i] beans. The array P is given as:
{P}

Alice and Bob play a game with the following rules:
- Alice goes first. They take turns alternately.
- On each turn, a player must choose three indices i, j, k (0 ≤ i < j ≤ k < {N}) such that the i-th bottle contains at least one bean. The player then removes one bean from bottle i, adds one bean to bottle j, and adds one bean to bottle k. (If j = k, it means adding two beans to bottle j.)
- The game ends when a player cannot make a move. The player who cannot move loses the game.

Assuming both players play optimally, who will win the game? Output a single line containing either `Alice` or `Bob` (do NOT include quotes or backticks)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the SplittingGame_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        self.parameter["reference_answer"] = "Alice" if random.random() < 0.5 else "Bob"


        def mex(s):
                m = 0
                while m in s:
                    m += 1
                return m
            
        # Precompute Sprague-Grundy values for reversed positions 0..N-1
        SG = [0] * N
        for r in range(1, N):
            reachable = set()
            for j in range(r):
                for k in range(j + 1):
                    reachable.add(SG[j] ^ SG[k])
            SG[r] = mex(reachable)

        while True :
            p = self.parameter["P"] = [random.randint(0, 2 * N) for _ in range(N)]
            
            def get_answer() :
                ans = 0
                # Compute nim-sum based on parity of beans
                for i in range(N):
                    if p[i] & 1:
                        r = N - 1 - i
                        ans ^= SG[r]

                # If zero nim-sum, losing position
                if ans == 0:
                    return "Bob"

                # Enumerate all valid moves i < j <= k with at least one bean at i
                for i in range(N):
                    if p[i] == 0:
                        continue
                    for j in range(i + 1, N):
                        for k in range(j, N):
                            r_i = N - 1 - i
                            r_j = N - 1 - j
                            r_k = N - 1 - k
                            # Check if this move leads to zero nim-sum
                            if ans ^ SG[r_i] ^ SG[r_j] ^ SG[r_k] == 0:
                                return "Alice"
                return "Bob"

            if get_answer() == self.parameter["reference_answer"] :
                break
    
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            P = " ".join("P[{}]={}".format(i, Pi) for i, Pi in enumerate(self.parameter["P"])),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            return answer.strip()
        else :
            return None
    
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result not in ("Alice", "Bob") :
                return self.rewards["invalid_answer"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]