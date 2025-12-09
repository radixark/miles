import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SecretCowCode_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3612
    prompt_template = \
r"""You are given a string S consisting of lowercase English letters: {S}
Define F(s) as the string obtained by concatenating `s` with `right_shift(s)` (s + right_shift(s)), where `right_shift(s)` means moving the last character of `s` to the beginning. Let F⁽∞⁾(S) denote the result of applying F infinitely many times to S: F⁽∞⁾(S) = F(F(F(...(S)...))). Please output the {K}-th character (1-based index, from left to right) of the infinite string F⁽∞⁾(S)."""
    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the SecretCowCode_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }


    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 2, "MAX_N should be greater than or equal to 2"
        
        S = self.parameter["S"] = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k = random.randint(2, MAX_N)))

        assert "MAX_K" in self.parameter, "MAX_K is required in parameter"
        MAX_K = self.parameter["MAX_K"]
        assert MAX_K > MAX_N, "MAX_K should be greater than MAX_N"
        K = self.parameter["K"] = random.randint(len(S) + 1, MAX_K)


        N = K

        # Build list of string lengths until covering N
        lengths = [len(S)]
        while lengths[-1] < N:
            lengths.append(lengths[-1] * 2)

        # Work backwards to map N into the original string
        while len(lengths) > 1:
            lengths.pop()
            half = lengths[-1]             # Length of the previous stage
            if N > half:
                if N == half + 1:
                    N = half
                else:
                    N = N - (half + 1)
            # if N <= half: it stays the same

        self.parameter["reference_answer"] = S[N-1]
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(S = self.parameter["S"], K = self.parameter["K"])
    

    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            return answer.strip()
        else :
            return None
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if len(processed_result) != 1 :
                return self.rewards["wrong_format"]
            if processed_result not in self.parameter["S"] :
                return self.rewards["wrong_format"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]