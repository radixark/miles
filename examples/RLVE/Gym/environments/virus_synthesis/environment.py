import random
from collections import deque
from typing import Optional
from Gym.environment import VerifiableEnvironment


class VirusSynthesis_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4762
    prompt_template = \
r"""Starting from an empty string, you can perform the following operations:
1. Add a single character to either the beginning or the end of the string.
2. Let the current string be S and its reverse be S'. You can append S' to either the beginning or the end of S (i.e., form S' + S or S + S', where + denotes string concatenation).

Your task is to obtain the target string by performing the minimum number of operations: {S}
**Output Format:** Output a single integer — the minimum number of operations required to construct the string given above."""

    def __init__(self,
                 wrong_format : float = -1.0, wrong_answer : float = 0.0, correct_answer : float = +1.0,
                 **kwargs) :
        """
        Initialize the VirusSynthesis_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_answer" : wrong_answer,
            "correct_answer" : correct_answer,
        }
    

    def _generate(self) -> None :
        assert "loose_MAX_N" in self.parameter, "loose_MAX_N is required in parameter"
        loose_MAX_N = self.parameter["loose_MAX_N"]
        assert loose_MAX_N >= 4, "loose_MAX_N should be greater than or equal to 4"

        operation_probabilities = [random.randint(1, loose_MAX_N) for _ in range(4)]
        operation_probabilities = [p / sum(operation_probabilities) for p in operation_probabilities]
        S = ""
        while True :
            operation = random.choices(population = ["1_beginning", "1_end", "2_beginning", "2_end"], weights = operation_probabilities)[0]
            if operation.startswith("1_") :
                char = random.choice("01")
                if operation == "1_beginning" :
                    S = char + S
                elif operation == "1_end" :
                    S = S + char
                else :
                    assert False
            elif operation.startswith("2_") :
                S_rev = S[:: -1]
                if operation == "2_beginning" :
                    S = S_rev + S
                elif operation == "2_end" :
                    S = S + S_rev
                else :
                    assert False
            else :
                assert False
            if len(S) >= loose_MAX_N :
                break
        self.parameter["S"] = S


        def min_operations(S):
            n = len(S)
            # Map nucleotides to indices
            char2idx = {'0': 0, '1': 1}
            # Palindromic tree structures
            ch = [[-1] * 4 for _ in range(2)]  # child pointers, -1 means absent
            fail = [1, 1]                      # fail links
            len_list = [0, -1]                 # palindrome lengths
            tran = [0, 0]                      # series links

            tot = 1    # current largest node index
            cur = 0    # current node (last added)

            def get_fail(x, pos):
                # Find the largest palindrome we can extend
                while pos - len_list[x] - 1 < 0 or S[pos - len_list[x] - 1] != S[pos]:
                    x = fail[x]
                return x

            # Build the palindromic tree
            for pos in range(n):
                c = char2idx[S[pos]]
                posx = get_fail(cur, pos)
                if ch[posx][c] == -1:
                    tot += 1
                    ch.append([-1] * 4)
                    len_list.append(len_list[posx] + 2)
                    # Compute fail link for the new node
                    f = get_fail(fail[posx], pos)
                    f2 = ch[f][c]
                    if f2 == -1:
                        f2 = 0
                    fail.append(f2)
                    # Compute series link (tran)
                    if len_list[tot] <= 2:
                        tran.append(f2)
                    else:
                        now = tran[posx]
                        while (pos - len_list[now] - 1 < 0 or
                            S[pos - len_list[now] - 1] != S[pos] or
                            (len_list[now] + 2) * 2 > len_list[tot]):
                            now = fail[now]
                        tran.append(ch[now][c])
                    # Link the new node
                    ch[posx][c] = tot
                cur = ch[posx][c]

            # DP over the palindromic tree to compute minimal operations
            dp = [0] * (tot + 1)
            for i in range(2, tot + 1):
                dp[i] = len_list[i]
            dp[0] = 1

            q = deque([0])
            ans = n
            while q:
                now = q.popleft()
                for c in range(4):
                    son = ch[now][c]
                    if son == -1:
                        continue
                    # Option 1: add one nucleotide
                    dp[son] = dp[now] + 1
                    # Option 2: copy-paste a palindrome
                    alt = dp[tran[son]] + 1 + len_list[son] // 2 - len_list[tran[son]]
                    if alt < dp[son]:
                        dp[son] = alt
                    # Combine with remaining suffix
                    cost = dp[son] + n - len_list[son]
                    if cost < ans:
                        ans = cost
                    q.append(son)
            return ans
        self.parameter["reference_answer"] = min_operations(S)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(S = self.parameter["S"])


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