import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Kth_SemiBalancedBracketSequence_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Consider strings that only contain the characters `(` and `)`:
- A string is called a **balanced bracket sequence** if, after inserting digits and operators, it can form a valid arithmetic expression. For example, `(())` is a balanced bracket sequence, while `)(()` is not.
- A string is called a **semi-balanced bracket sequence** if removing **exactly one bracket** from it can result in a balanced bracket sequence.

We define the lexicographical order such that `(` comes **before** `)`. Please find the **{K}-th semi-balanced bracket sequence of length {N}**, when all such sequences are sorted in lexicographical order.

**Output Format:** Your final answer should be a single line containing the semi-balanced bracket sequence."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 5.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the Kth_SemiBalancedBracketSequence_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"
        assert N % 2 == 1, "N should be odd"

        cbs = [[0] * (N + 2) for _ in range(N + 2)]
        cbs[0][0] = 1
        for i in range(1, N + 1) :
            cbs[i][0] = cbs[i - 1][1]
            for j in range(1, N + 1) :
                cbs[i][j] = cbs[i - 1][j - 1] + cbs[i - 1][j + 1]

        total = 0
        for i in range(0, N + 1, 2) :
            total += 2 * cbs[i][0] * cbs[N - 1 - i][0]
        
        K = self.parameter["K"] = random.randint(1, total)


        K -= 1

        s = ["("] * N
        b = [0] * (N + 2)
        good = [[False] * (N + 2) for _ in range(N + 2)]
        for i in range(1, N + 2) :
            good[i][i - 1] = True

        for i in range(1, N + 1) :
            b[i] = b[i - 1] + 1
            for j in range(1, i + 1) :
                good[j][i] = good[j][i - 1] and (b[i] - b[j - 1] >= 0)

            cur = 0
            for j in range(1, i + 1) :
                if good[1][j - 1] and b[j - 1] == 0 and good[j + 1][i] :
                    cur += cbs[N - i][b[i] - b[j]]
            if good[1][i] :
                for j in range(i + 1, N + 1) :
                    cur += 2 * cbs[j - i - 1][b[i]] * cbs[N - j][0]

            if cur <= K :
                K -= cur
                s[i - 1] = ")"
                b[i] = b[i - 1] - 1
                for j in range(1, i + 1) :
                    good[j][i] = good[j][i - 1] and (b[i] - b[j - 1] >= 0)
        
        assert len(s) == N and all([c in "()" for c in s]), "The generated sequence is not valid"
        self.parameter["reference_answer"] = "".join(s)
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], K = self.parameter["K"])
    

    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            answer = answer.strip()
            return answer
        else :
            return None
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if not (len(processed_result) == self.parameter["N"] and all(c in "()" for c in processed_result)) :
                return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(float(a == b) for a, b in zip(self.parameter["reference_answer"], processed_result)) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]