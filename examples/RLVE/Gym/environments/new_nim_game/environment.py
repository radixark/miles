import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class NewNimGame_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4301
    prompt_template = \
r"""You are given a Nim-like game with heaps of matches. There are {N} heaps with the following sizes (1-indexed): {A}
Game rules:
- **First round** has two phases:
  1) **Your move (first player):** You may remove **any number of entire heaps** (possibly zero), but you are **not allowed** to remove **all** heaps.
  2) **Opponent's move (second player):** Then the opponent may remove **any number of entire heaps** (possibly zero), but likewise cannot remove **all remaining** heaps.
- **From the second round onward:** Standard Nim rules apply on the remaining heaps: players alternate; a move removes any positive number of matches from **exactly one** heap; the player who takes the last match **wins**.
- Both players play optimally.

Your task: Choose which heaps to remove **in your first move** so that you **guarantee a win**; if multiple winning choices exist, choose one that **minimizes the total number of matches** you remove (i.e., the sum of sizes of the heaps you remove). Output the distinct *indices** (1-based) of the heaps you remove in your first move, in any order, separated by spaces; if you can guarantee victory without removing any heap, output an **empty line**."""

    def __init__(self,
                 match_number_range_coefficient : int = 2,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = +3.0,
                 **kwargs) :
        """
        Initialize the NewNimGame_Environment instance.
        """
        super().__init__(**kwargs)

        self.match_number_range_coefficient = match_number_range_coefficient
        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "unsuccessful_solution" : unsuccessful_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N must be at least 3"

        self.parameter["A"] = [random.randint(1, N * self.match_number_range_coefficient) for i in range(N)]


        A = self.parameter["A"].copy()
        A.sort(reverse=True)

        max_bit = max(A).bit_length()
        D = [0] * max_bit  # linear basis, dynamic size based on input
        ans = 0

        def add(x):
            # Try to insert x into the xor-basis D
            for i in range(max_bit - 1, -1, -1):
                if (x >> i) & 1:
                    if D[i]:
                        x ^= D[i]
                    else:
                        D[i] = x
                        return True
            return False
        
        for x in A:
            if not add(x):
                ans += x

        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = ", ".join("the size of heap {} is {}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"], start = 1)),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List[int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            if len(processed_result) != len(set(processed_result)) :
                return self.rewards["invalid_solution"] # Duplicate indices
            if not all(1 <= index <= self.parameter["N"] for index in processed_result) :
                return self.rewards["invalid_solution"] # Index out of range
            if len(processed_result) == self.parameter["N"] :
                return self.rewards["invalid_solution"] # Cannot remove all heaps
            
            removed = [False] * self.parameter["N"]
            for index in processed_result :
                removed[index - 1] = True
            
            max_bit = max(self.parameter["A"]).bit_length()
            D = [0] * max_bit  # linear basis, dynamic size based on input
            def add(x):
                # Try to insert x into the xor-basis D
                for i in range(max_bit - 1, -1, -1):
                    if (x >> i) & 1:
                        if D[i]:
                            x ^= D[i]
                        else:
                            D[i] = x
                            return True
                return False
            for i, Ai in enumerate(self.parameter["A"]) :
                if not removed[i] :
                    if not add(Ai):
                        return self.rewards["unsuccessful_solution"] # Cannot guarantee victory
            
            answer, gold = sum(self.parameter["A"][i - 1] for i in processed_result), self.parameter["gold_answer"]
            assert 0 <= gold <= answer, "Gold answer should be non-negative and not exceed the provided answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if answer == 0 :
                    assert gold == 0, "If answer is 0, gold must also be 0"
                    return self.rewards["rewarding_weight"] * 1.0
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]