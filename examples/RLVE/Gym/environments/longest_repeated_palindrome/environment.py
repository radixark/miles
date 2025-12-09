import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Longest_RepeatedPalindrome_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a string S: {S}

Please find a **substring T of S** such that:
- T = A + A^R + A + A^R, where A^R denotes the reverse of string A, and + represents string concatenation.
- Try your best to **maximize the length** of T.

**Output Format:** Output a single line containing the substring T."""

    def __init__(self,
                 wrong_format: float = -1.0, invalid_solution: float = -0.5, rewarding_strategy: str = "(answer/gold)^beta", rewarding_weight: float = +1.0, rewarding_beta: float = 5.0,
                 **kwargs):
        """
        Initialize the Longest_RepeatedPalindrome_Environment instance.
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
        assert N >= 4, "N should be greater than or equal to 4"

        one_probability = random.uniform(0.1, 0.9)

        A_length = random.randint(1, N // 4)
        A = "".join("1" if random.random() < one_probability else "0" for _ in range(A_length))
        A_reverse = A[::-1]
        first_length = random.randint(0, N - 4 * A_length)
        first_part = "".join("1" if random.random() < one_probability else "0" for _ in range(first_length))
        second_length = N - first_length - 4 * A_length
        second_part = "".join("1" if random.random() < one_probability else "0" for _ in range(second_length))
        S = self.parameter["S"] = first_part + A + A_reverse + A + A_reverse + second_part
        assert len(S) == N, "S should have length N"


        def compute(S):
            n = len(S)
            
            # Prepare for Palindromic Tree (PAM)
            # We use two root nodes: node 0 for even-length palindromes (length 0)
            # and node 1 for odd-length root (length -1).
            # Maximum number of nodes is at most n + 3.
            size = n + 3
            ch = [[0] * 2 for _ in range(size)]  # transitions
            fail = [0] * size                     # failure links
            f = [0] * size                        # auxiliary links for double palindrome
            length = [0] * size                   # palindrome lengths

            tot = 1          # total nodes so far (we have nodes 0 and 1)
            fail[0] = 1      # fail of even root -> odd root
            length[1] = -1   # length of odd root
            las = 0          # last added node (start at even root)

            # Shift string to 1-indexed for convenience
            S = ' ' + S

            for i in range(1, n + 1):
                cur = las
                # Find the largest suffix-palindrome we can extend
                while S[i] != S[i - length[cur] - 1]:
                    cur = fail[cur]
                c = int(S[i])

                # If this extension hasn't been created, build a new node
                if ch[cur][c] == 0:
                    tot += 1
                    length[tot] = length[cur] + 2

                    # Compute failure link for the new node
                    x = fail[cur]
                    while S[i] != S[i - length[x] - 1]:
                        x = fail[x]
                    fail[tot] = ch[x][c]

                    ch[cur][c] = tot

                    # Compute auxiliary link f for checking double palindrome
                    if length[fail[tot]] <= length[tot] // 2:
                        f[tot] = fail[tot]
                    else:
                        p = f[cur]
                        # Traverse until we find a valid half-length palindrome to extend
                        while (length[p] + 2 > length[tot] // 2) or (S[i] != S[i - length[p] - 1]):
                            p = fail[p]
                        f[tot] = ch[p][c]

                # Move last pointer
                las = ch[cur][c]

            # Compute the answer: longest double palindrome length
            ans = 0
            # Nodes start from index 2 (skip the two roots)
            for i in range(2, tot + 1):
                if length[i] % 4 == 0 and length[f[i]] == length[i] // 2:
                    ans = max(ans, length[i])
            return ans
        self.parameter["gold_answer"] = compute(S)
        assert self.parameter["gold_answer"] >= 4 * A_length, "gold_answer should be at least 4 * A_length"
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(S = self.parameter["S"])


    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            return answer.strip()
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            T = processed_result
            if T not in self.parameter["S"] :
                return self.rewards["invalid_solution"]
            if len(T) == 0 or len(T) % 4 != 0 :
                return self.rewards["invalid_solution"]
            A = T[: len(T) // 4]
            A_reverse = A[::-1]
            if T != A + A_reverse + A + A_reverse :
                return self.rewards["invalid_solution"]
            
            answer, gold = len(T), self.parameter["gold_answer"]
            assert answer <= gold, "Answer should not be greater than gold answer"
            
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * int(answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]