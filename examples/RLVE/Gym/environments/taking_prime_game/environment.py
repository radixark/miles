import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class TakingPrimeGame_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1857
    prompt_template = \
r"""There are {N} stones in a pile and two players: Stan and his opponent. On each turn, a player may remove any **prime number** of stones from the pile. A player who cannot make a move **loses** the game.

Stan goes first. Both players play **optimally**:
- If a player is guaranteed to win, they will try to win in the **minimum number of moves** possible.
- If a player is guaranteed to lose, they will try to **delay the loss** as much as possible.

**Output Format:**
Your final answer should be a single integer:
- The **total number of moves** (both players’) until Stan wins (if he must win), or
- `-1` (if he must lose).
Do **NOT** include quotes or backticks."""

    def __init__(self,
                 lose_probability : float = 0.2,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the TakingPrimeGame_Environment instance.
        """
        super().__init__(**kwargs)

        self.lose_probability = lose_probability
        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    
    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 1, "MAX_N should be greater than or equal to 1"


        is_prime = [True] * (MAX_N + 1)
        if MAX_N >= 0 :
            is_prime[0] = False
        if MAX_N >= 1 :
            is_prime[1] = False
        primes = []
        for i in range(2, MAX_N + 1) :
            if is_prime[i] :
                primes.append(i)
                for j in range(i * i, MAX_N + 1, i) :
                    is_prime[j] = False

        win = [False] * (MAX_N + 1)
        dp_moves = [0] * (MAX_N + 1)

        for i in range(2, MAX_N + 1) :
            min_moves = (MAX_N + 1) * 100
            max_moves = 0
            has_winning_move = False
            for p in primes :
                if p > i :
                    break
                if not win[i - p] :
                    has_winning_move = True
                    min_moves = min(min_moves, dp_moves[i - p] + 1)
                else :
                    max_moves = max(max_moves, dp_moves[i - p] + 1)
            if has_winning_move :
                win[i] = True
                dp_moves[i] = min_moves
            else :
                win[i] = False
                dp_moves[i] = max_moves
        
        lose = random.random() < self.lose_probability
        while True :
            N = self.parameter["N"] = random.randint(1, MAX_N)
            if win[N] != lose :
                break
        self.parameter["reference_answer"] = dp_moves[N] if win[N] else -1
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"])


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