import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class KthSubsequence_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3728
    prompt_template = \
r"""You are given a string S of length {N}: {S}
There are 2^{N} - 1 non-empty subsequences of S (a subsequence is a string obtained by deleting some characters of S without changing the order of the remaining characters; for example, "abc" is a subsequence of "aebdc"). Among all these subsequences, keep only the **unique** ones and sort them in **lexicographical order**. Number them starting from 1. Please find the {K}-th string in this sorted list.

**Output Format:** A single string — the {K}-th unique subsequence of S in lexicographical order."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the KthSubsequence_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        a_probability = random.random()
        S = self.parameter["S"] = "".join("a" if random.random() < a_probability else "b" for _ in range(N))
        assert len(S) == N, "Generated string S does not match the specified length N"

        Next = [[None] * 2 for i in range(N)]
        F = [0] * N
        for i in range(N - 1, -1, -1) :
            Si = ord(S[i]) - ord('a')
            F[i] = 1
            for c in range(2) :
                Next[i][c] = Next[i + 1][c] if i + 1 < N else None
                if c == Si :
                    Next[i][c] = i
            if i + 1 < N :
                for c in range(2) :
                    if Next[i + 1][c] is not None :
                        F[i] += F[Next[i + 1][c]]
        K = 0
        for c in range(2) :
            if Next[0][c] is not None :
                K += F[Next[0][c]]
        K = self.parameter["K"] = random.randint(1, K)


        def compute(K : int) -> str :
            result = ""
            index = 0
            while True :
                assert 0 <= index < N, "Index out of bounds"
                found = False
                for c in range(26) :
                    if Next[index][c] is not None :
                        if F[Next[index][c]] >= K :
                            result += chr(c + ord('a'))
                            if K == 1 :
                                return result
                            else :
                                index = Next[index][c] + 1
                                K -= 1
                                found = True
                                break
                        else :
                            K -= F[Next[index][c]]
                assert found, "No valid character found, this should not happen"
        self.parameter["reference_answer"] = compute(K)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], S = self.parameter["S"], K = self.parameter["K"])
    

    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            return answer.strip()
        else :
            return None
    
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if any(c not in "ab" for c in processed_result) :
                return self.rewards["wrong_format"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]