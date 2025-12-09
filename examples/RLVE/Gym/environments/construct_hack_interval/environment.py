import random
from typing import Optional, Tuple
from Gym.environment import VerifiableEnvironment

class ConstructHackInterval_Environment(VerifiableEnvironment) : # Source: https://codeforces.com/problemset/problem/468/C
    prompt_template = \
r"""Let's define f(x) as the sum of digits in the decimal representation of number x (for example, f(1234) = 1 + 2 + 3 + 4). Please construct an interval [L, R], such that the sum of f(x) for all x in the interval is divisible by {MOD}.
Note that L and R should be both positive integers, L should be less than or equal to R, and R should be less than or equal to 10 * {MOD}.

Output Format: Your final answer should be **two integers** on a line by itself, representing the value of L and R of the interval.  Example: `5 123` (do **NOT** include the backticks or quotes).
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, correct_solution : float = +1.0, wrong_solution : float = 0.0,
                 **kwargs) :
        """
        Initialize the ConstructHackInterval_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "correct_solution" : correct_solution,
            "wrong_solution" : wrong_solution,
        }
    
    def _generate(self) -> None :
        assert "MAX_MOD" in self.parameter, "MAX_MOD is required in parameter"
        MAX_MOD = self.parameter["MAX_MOD"]
        assert MAX_MOD >= 1, "MAX_MOD should be greater than or equal to 1"
        
        MOD = self.parameter["MOD"] = random.randint(1, MAX_MOD)
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            MOD = self.parameter["MOD"],
        )

    def _process(self, answer : Optional[str]) -> Optional[Tuple[int, int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                L, R = map(int, answer.split())
                return L, R
            except :
                return None
        else :
            return None
    
    def count_digit_sum(self, L, R):
        def count_digits_up_to(n):
            '''
            Count the sum of digits of all numbers in the interval [0, n].
            '''
            if n < 0:
                return 0
            if n < 10:
                return sum(range(1, n + 1))
                
            # Count digits in numbers up to n
            digits = len(str(n))
            total = 0
            first_digit = int(str(n)[0])
            remaining = int(str(n)[1:]) if len(str(n)) > 1 else 0
            
            # Count digits in numbers with fewer digits: 00..0 to (x-1)99..9 (d-1 full digits, x is the first digit)
            total += (digits-1) * 45 * (10 ** (digits-2)) * first_digit + first_digit * (first_digit - 1) // 2 * (10 ** (digits-1))
            
            # Add contribution from remaining part: >= x00..0 to n
            total += count_digits_up_to(remaining) + first_digit * (remaining + 1)
            
            return total
            
        return count_digits_up_to(R) - count_digits_up_to(L-1)

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            L, R = processed_result
            if not (1 <= L <= R and R <= 10 * self.parameter["MOD"]) :
                return self.rewards["invalid_solution"]
            digit_sum = self.count_digit_sum(L, R)
            MOD = self.parameter["MOD"]
            if digit_sum % MOD == 0 :
                return self.rewards["correct_solution"]
            else :
                return self.rewards["wrong_solution"]
        else :
            return self.rewards["wrong_format"]