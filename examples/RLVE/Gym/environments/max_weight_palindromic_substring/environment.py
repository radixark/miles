import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MaxWeightPalindromicSubstring_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3649
    prompt_template = \
r"""You are given a string S: {S}
Please find a palindromic string T such that the product of T's length and the number of times T occurs in S is maximized.
**Output Format:** Output a single line containing the string T."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MaxWeightPalindromicSubstring_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        a_probability = random.uniform(0.3, 0.7)
        S = self.parameter["S"] = "".join("a" if random.random() < a_probability else "b" for _ in range(N))


        def max_palindrome_existence_value(S: str) -> int:
            """
            Build a palindromic tree (Eertree) for S and compute the maximum
            existence value among all palindromic substrings: length * frequency.
            """
            N = len(S)
            # We will have at most N+2 distinct palindromes plus two roots
            size = 1  # the last-used node index
            # length of palindrome at each node
            length = [0] * (N + 3)
            # failure link (longest proper palindromic suffix) for each node
            fail = [0] * (N + 3)
            # count of how many times this palindrome occurs as a suffix during construction
            count = [0] * (N + 3)
            # transitions: for each node, map character -> next node
            trans = [dict() for _ in range(N + 3)]

            # two roots:
            # node 1: imaginary palindrome of length -1
            # node 0: empty palindrome of length 0
            length[1] = -1
            fail[0] = 1
            fail[1] = 1

            last = 0  # the node corresponding to the longest palindromic suffix of S[:i]

            for i, c in enumerate(S):
                cur = last
                # find the longest suffix-palindrome of S[:i] that we can extend by c
                while True:
                    if i - length[cur] - 1 >= 0 and S[i - length[cur] - 1] == c:
                        break
                    cur = fail[cur]

                # if there is no outgoing edge for c, create a new node
                if c not in trans[cur]:
                    size += 1
                    length[size] = length[cur] + 2

                    # compute failure link for the new node
                    f = fail[cur]
                    while True:
                        if i - length[f] - 1 >= 0 and S[i - length[f] - 1] == c:
                            break
                        f = fail[f]
                    # may be 0 if it's the first single-character palindrome
                    fail[size] = trans[f].get(c, 0)

                    # link cur --c--> size
                    trans[cur][c] = size

                # move to that node and count one occurrence
                last = trans[cur][c]
                count[last] += 1

            # propagate the counts from longer palindromes to their suffix-palindromes
            ans = 0
            for u in range(size, 1, -1):
                # existence value = length[u] * total occurrences of this palindrome
                ans = max(ans, length[u] * count[u])
                count[fail[u]] += count[u]

            assert ans > 0
            return ans
        self.parameter["gold_answer"] = max_palindrome_existence_value(S)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(S = self.parameter["S"])


    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            answer = answer.strip()
            return answer
        else :
            return None
    
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result != processed_result[::-1] :
                return self.rewards["invalid_solution"]
            
            def count_overlapping_occurrences_kmp(text, pattern):
                if not pattern or not text:
                    return 0
                def build_failure_function(pattern):
                    m = len(pattern)
                    failure = [0] * m
                    j = 0
                    
                    for i in range(1, m):
                        while j > 0 and pattern[i] != pattern[j]:
                            j = failure[j - 1]
                        if pattern[i] == pattern[j]:
                            j += 1
                        failure[i] = j
                    return failure
                failure = build_failure_function(pattern)
                count = 0
                j = 0
                for i in range(len(text)):
                    while j > 0 and text[i] != pattern[j]:
                        j = failure[j - 1]
                    if text[i] == pattern[j]:
                        j += 1
                    if j == len(pattern):
                        count += 1
                        j = failure[j - 1]
                return count
            
            answer, gold = len(processed_result) * count_overlapping_occurrences_kmp(self.parameter["S"], processed_result), self.parameter["gold_answer"]
            assert answer <= gold
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]