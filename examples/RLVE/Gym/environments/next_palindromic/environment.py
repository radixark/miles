import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class NextPalindromic_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1609
    prompt_template = r"""Please find the **smallest palindromic number** that is greater than {N}."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the NextPalindromic_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "digit_num" in self.parameter, "digit_num is required in parameter"
        digit_num = self.parameter["digit_num"]
        assert digit_num >= 1, "digit_num should be greater than or equal to 1"

        self.parameter["N"] = random.randint(1, 10 ** digit_num - 1)


        def next_palindrome(s: str) -> str:
            l = len(s)
            # Special case: all '9's -> next palindrome is 1 followed by zeros and ending with 1
            if all(ch == '9' for ch in s):
                return '1' + '0' * (l - 1) + '1'

            # Build initial palindrome by mirroring left half to right half
            ans = list(s)
            for i in range(l // 2):
                ans[l - 1 - i] = ans[i]

            # If this palindrome is already greater than the original, return it
            if ''.join(ans) > s:
                return ''.join(ans)

            # Otherwise, increment the middle and propagate carry
            # Find the middle index (for both even and odd lengths)
            mid = (l - 1) // 2
            i = mid
            # Move left through the middle until a non-'9' digit is found, setting '9's to '0'
            while i >= 0 and ans[i] == '9':
                ans[i] = '0'
                i -= 1
            # Increment the first non-'9' digit
            ans[i] = str(int(ans[i]) + 1)
            # Mirror the incremented digit to the other side
            ans[l - 1 - i] = ans[i]

            # Mirror the rest of the left half to the right half to form a valid palindrome
            for j in range(l // 2):
                ans[l - 1 - j] = ans[j]

            return ''.join(ans)
        self.parameter["reference_answer"] = next_palindrome(str(self.parameter["N"]))
    
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
            if not (processed_result > self.parameter["N"]) :
                return self.rewards["invalid_solution"]
            if str(processed_result) != str(processed_result)[::-1] :
                return self.rewards["invalid_solution"]

            gold, answer = int(self.parameter["reference_answer"]), processed_result
            assert gold <= answer

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]