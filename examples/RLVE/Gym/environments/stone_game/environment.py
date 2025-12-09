import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class StoneGame_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3235
    prompt_template = \
r"""Stan and Ollie are playing a game. The game rules are as follows:
+ There are **{N}** heaps of stones: {Stones}.
+ Stan and Ollie take turns playing, and **Stan** goes first.
+ On a player's turn, they must select a heap that contains at least **{F}** stones.
+ Then, they choose an integer **M** (at least 2) and split the selected heap into **M** smaller heaps such that the sizes of the smaller heaps differ by at most 1 (i.e., as evenly as possible).
+ After splitting, the game continues with the updated heap configuration.
+ If a player cannot make a move (i.e., no heap contains at least **{F}** stones), they lose.

If both players always play optimally, who will win — Stan or Ollie?

**Output Format:** Your final answer should be a single word: either `Stan` or `Ollie` (do **NOT** include quotes or backticks), indicating the winner."""
    
    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the StoneGame_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    
    def _generate(self) -> None :
        assert "MAX_SUM" in self.parameter, "MAX_SUM is required in parameter"
        MAX_SUM = self.parameter["MAX_SUM"]
        assert MAX_SUM >= 2, "MAX_SUM should be greater than or equal to 2"

        self.parameter["reference_answer"] = "Stan" if random.random() < 0.5 else "Ollie"

        while True :
            SUM = random.randint(2, MAX_SUM)
            N = self.parameter["N"] = random.randint(1, min(SUM // 2, 100))
            if N == 1:
                Stones = [SUM]
            else:
                cuts = sorted(random.sample(range(1, SUM), N - 1))
                Stones = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, N - 1)] + [SUM - cuts[-1]]
            self.parameter["Stones"] = Stones
            F = self.parameter["F"] = random.randint(1, max(Stones) + 1)
            
            def check(n : int, f : int, stones : List[int]) -> bool :
                sg = [-1] * (max(stones) + 5)
                exist = [0] * (max(stones) + 5)
                for i in range(0, min(max(stones)+1, f)):
                    sg[i] = 0
                
                def get_sg(x):
                    if sg[x] != -1: return sg[x] 
                    i = 2
                    while i <= x :
                        k = x//(x//i)
                        for j in range(i, min(i+1, k)+1):
                            s = 0
                            if (x%j) % 2 == 1: s ^= get_sg(x//j+1)
                            if (j-(x%j)) % 2 == 1: s ^= get_sg(x//j)
                            exist[s] = x 
                        i = k + 1
                    i = 0
                    while True:
                        if exist[i] != x: 
                            sg[x] = i 
                            return i
                        i += 1
                
                nim_sum = 0
                for pile_size in stones:
                    nim_sum ^= get_sg(pile_size)
                return nim_sum != 0
            
            if ("Stan" if check(N, F, Stones) else "Ollie") == self.parameter["reference_answer"] :
                break
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], F = self.parameter["F"], Stones = ", ".join(map(str, self.parameter["Stones"])))


    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            return answer.strip()
        else :
            return None
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result not in ("Stan", "Ollie") :
                return self.rewards["invalid_answer"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]