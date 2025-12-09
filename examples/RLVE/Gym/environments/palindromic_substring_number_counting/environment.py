import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class PalindromicSubstringNumberCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3413
    prompt_template = \
r"""We treat every positive integer as a string of digits (without leading zeros). A number is called a `good number` if it contains at least one palindromic substring of length **greater than 1**.

For example:
- 101 is a good number because it contains the substring "101",
- 110 is a good number because it contains the substring "11",
- But 102 and 1201 are not good numbers because they do not contain any palindromic substring of length greater than 1.

Please count how many good numbers exist in the range [{L}, {R}] (inclusive)."""


    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the PalindromicSubstringNumberCounting problem.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_R" in self.parameter, "MAX_R is required in parameter"
        MAX_R = self.parameter["MAX_R"]
        assert MAX_R >= 20, "MAX_R should be greater than or equal to 20"

        R = self.parameter["R"] = random.randint(20, MAX_R)
        L = self.parameter["L"] = random.randint(1, R - 1)


        def str_minus_one(s: str) -> str:
            # Subtract 1 from a positive decimal string s
            lst = list(s)
            i = len(lst) - 1
            # borrow until we find a non-zero digit
            while i >= 0 and lst[i] == '0':
                lst[i] = '9'
                i -= 1
            if i >= 0:
                lst[i] = str(int(lst[i]) - 1)
            # strip leading zeros (but leave one zero if result is 0)
            if lst[0] == '0':
                j = 0
                while j < len(lst) - 1 and lst[j] == '0':
                    j += 1
                lst = lst[j:]
            return ''.join(lst)

        def solve_for(bound_str: str) -> int:
            # Count "lovely" numbers in [0, bound_str]
            n = len(bound_str)
            # d[1] = least significant digit, ..., d[n] = most significant
            d = [0] * (n + 1)
            for i, ch in enumerate(reversed(bound_str), start=1):
                d[i] = int(ch)

            # dp cache: f[x][num][pre][lovely][lead][prelead], initialized to -1
            f = [[[[[[ -1 for _ in range(2)] 
                        for _ in range(2)] 
                        for _ in range(2)] 
                    for _ in range(10)] 
                    for _ in range(10)] 
                    for _ in range(n+1)]

            def dfs(x: int, num: int, pre: int, lovely: bool,
                    lead: bool, prelead: bool, top: bool) -> int:
                # base case: all digits placed
                if x == 0:
                    return 1 if lovely else 0

                # use cache when not tight
                if not top:
                    cached = f[x][num][pre][lovely][lead][prelead]
                    if cached != -1:
                        return cached

                bound = d[x] if top else 9
                total = 0

                for digit in range(bound + 1):
                    # check for palindrome substrings of length 2 or 3
                    is_lovely = lovely \
                        or ((not lead) and digit == num) \
                        or ((not prelead) and digit == pre)
                    next_lead = lead and (digit == 0)
                    next_prelead = lead
                    next_top = top and (digit == bound)

                    total += dfs(x - 1, digit, num,
                                        is_lovely, next_lead,
                                        next_prelead, next_top)

                if not top:
                    f[x][num][pre][lovely][lead][prelead] = total

                return total

            # start from position n, with no previous digits placed
            return dfs(n, 0, 0, False, True, True, True)

        # compute counts up to R and up to L-1, then take difference
        L, R = str(L), str(R)
        L_minus_one = str_minus_one(L)
        self.parameter["reference_answer"] = solve_for(R) - solve_for(L_minus_one)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(L = self.parameter["L"], R = self.parameter["R"])


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
            if processed_result < 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                if processed_result == 0 :
                    return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == 0)
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]