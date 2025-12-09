import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class DeltaNimGame_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3480
    prompt_template = \
r"""Alice and Bob are playing a game with {N} piles of stones. The number of stones in the i-th pile is A[i], for 0 <= i < {N}. The initial array A is: {A}

Game rules:
- Players alternate turns, with Alice going first.
- On a turn, a player chooses a pile `i` (0 <= i < {N}) and removes any number of stones (at least 1 and at most A[i]). After the move, the array A must still satisfy the condition: A[i] <= A[i + 1] for all 0 <= i < {N} - 1.
- A player who cannot make a valid move loses.

Assuming both players play optimally, determine who will win. Output a single word: `Alice` or `Bob` (do NOT include quotes or backticks), indicating the winner."""


    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the DeltaNimGame instance.
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
        assert N >= 2, "N should be greater than or equal to 2"

        self.parameter["reference_answer"] = "Alice" if random.random() < 0.5 else "Bob"

        C = [None] * N
        ans = 0
        for i in range(N) :
            if i != N - 1 :
                C[i] = random.randint(1 if i == 0 else 0, N)
            else :
                if self.parameter["reference_answer"] == "Alice" :
                    while True :
                        C[i] = random.randint(0, N)
                        if (ans ^ C[i]) != 0 :
                            break
                elif self.parameter["reference_answer"] == "Bob" :
                    C[i] = ans
                else :
                    assert False, "Invalid reference answer"
            if (i & 1) == ((N - 1) & 1) :
                ans ^= C[i]
        assert (ans == 0) == (self.parameter["reference_answer"] == "Bob"), "Reference answer does not match computed answer"
    
        A = self.parameter["A"] = [None] * N
        for i in range(N) :
            A[i] = (A[i - 1] if i - 1 >= 0 else 0) + C[i]
            if i >= 1 :
                assert A[i] >= A[i - 1], "A should be non-decreasing"
        assert A[0] >= 1
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
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