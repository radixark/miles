import random
from typing import Optional, Tuple
from Gym.environment import VerifiableEnvironment


class Longest_DoublePalindrome_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a string S of length {N} (0-indexed): {S}

Please find two non-empty intervals [A, B) and [B, C) (obviously, 0 <= A < B < C <= {N}) such that:
- S[A : B] and S[B : C] are both palindromes (S[a : b] refers to the substring starting at index a and ending at index b - 1, i.e., S[a] + S[a+1] + ... + S[b-1]).
- Try your best to maximize C - A.

**Output Format:** Your final answer should be three integers A, B, and C, separated by spaces."""

    def __init__(self,
                 wrong_format: float = -1.0, invalid_solution: float = -0.5, rewarding_strategy: str = "(answer/gold)^beta", rewarding_weight: float = +1.0, rewarding_beta: float = 5.0,
                 **kwargs):
        """
        Initialize the Longest_DoublePalindrome_Environment instance.
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
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        one_probability = random.uniform(0.1, 0.9)
        endpoints = random.sample(range(N + 1), 3)
        endpoints.sort()

        def generate_random(length : int) -> str :
            assert length >= 0, "length should be non-negative"
            return "".join("1" if random.random() < one_probability else "0" for _ in range(length))
        def generate_palindrome(length : int) -> str :
            assert length >= 1, "length should be at least 1"
            half = length // 2
            first_half = "".join("1" if random.random() < one_probability else "0" for _ in range(half))
            if length % 2 == 0:
                return first_half + first_half[::-1]
            else:
                return first_half + ("1" if random.random() < one_probability else "0") + first_half[::-1]
        S = self.parameter["S"] = \
            generate_random(endpoints[0]) + \
            generate_palindrome(endpoints[1] - endpoints[0]) + \
            generate_palindrome(endpoints[2] - endpoints[1]) + \
            generate_random(N - endpoints[2])
        assert len(S) == N, "S should have length N"


        modified = ['@', '#']
        for ch in S:
            modified.append(ch)
            modified.append('#')
        modified.append('$')
        M = len(modified)

        # Arrays for Manacher
        p = [0] * M
        # Arrays to record max palindromic radii ending/starting at positions
        l = [0] * M
        r = [0] * M

        center = 0
        right = 0

        # Manacher's algorithm on the modified string
        for i in range(1, M - 1):
            mirror = 2 * center - i
            if i < right:
                p[i] = min(right - i, p[mirror])
            # Expand around center i
            while modified[i + 1 + p[i]] == modified[i - 1 - p[i]]:
                p[i] += 1
            # Update center and right boundary
            if i + p[i] > right:
                center = i
                right = i + p[i]
            # Record palindromic spans (adjusted from C++ p: p_python = p_cpp - 1)
            if p[i] > 0:
                l[i + p[i]] = max(l[i + p[i]], p[i])
                r[i - p[i]] = max(r[i - p[i]], p[i])

        # Propagate the best spans outward
        # For l: propagate from right to left on odd indices
        for i in range(M - 4, 0, -2):
            l[i] = max(l[i], l[i + 2] - 2)
        # For r: propagate from left to right on odd indices
        for i in range(3, M, 2):
            r[i] = max(r[i], r[i - 2] - 2)

        # Compute the answer by checking split points at separator positions
        ans = 0
        for i in range(1, M, 2):  # only consider '#' positions
            if l[i] > 0 and r[i] > 0:
                ans = max(ans, l[i] + r[i])

        self.parameter["gold_answer"] = ans
        assert self.parameter["gold_answer"] >= endpoints[2] - endpoints[0]
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], S = self.parameter["S"])
    

    def _process(self, answer : Optional[str]) -> Optional[Tuple[int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                A, B, C = map(int, answer.split())
                return A, B, C
            except :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            A, B, C = processed_result
            if not (0 <= A < B < C <= self.parameter["N"]) :
                return self.rewards["invalid_solution"]
            def check_palindrome(s : str) -> bool :
                return s == s[:: -1]
            if not (check_palindrome(self.parameter["S"][A : B]) and check_palindrome(self.parameter["S"][B : C])) :
                return self.rewards["invalid_solution"]
            
            answer, gold = C - A, self.parameter["gold_answer"]
            assert answer <= gold, "answer should not be greater than gold"
            
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * int(answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]