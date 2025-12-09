import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class AntiPalindromicSubstringCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3501
    prompt_template = \
r"""We define an **anti-palindromic binary string** as a binary string such that its reverse is equal to the bitwise complement of the original string (i.e., '0' becomes '1' and '1' becomes '0'). For example, `000111` is anti-palindromic because its reverse is `111000`, which is the bitwise complement of `000111`. But `1001` is not, because its reverse is `1001`, while its flipped version is `0110`.

You are given a binary string: {S}
Please count the number of **contiguous substrings** of `S` that are anti-palindromic. Two substrings are considered different if they appear at different positions in `S`. Output a single integer — the number of anti-palindromic substrings."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the AntiPalindromicSubstringCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"
        
        endpoints = random.sample(range(1, N), random.randint(0, N - 1))
        endpoints.sort()
        endpoints = [0] + endpoints + [N]

        one_probability = random.random()
        
        S = ""
        for i in range(len(endpoints) - 1) :
            length = endpoints[i + 1] - endpoints[i]
            if length % 2 == 0 :
                half = "".join("1" if random.random() < one_probability else "0" for _ in range(length // 2))
                S += half + "".join("1" if c == "0" else "0" for c in reversed(half))
            else :
                S += "".join("1" if random.random() < one_probability else "0" for _ in range(length))
        self.parameter["S"] = S
        assert len(S) == N, f"Generated string length {len(S)} does not match N {N}"


        # Build the “S” array from the C++:
        #   S[0] = '$', S[1] = '#', then for each char: c, '#', and finally a trailing '$'
        T = ['$','#']
        for c in S:
            T.append(c)
            T.append('#')
        T.append('$')

        length = len(T)
        tot = length - 2   # corresponds to C++ `tot` (1 + 2*N)

        # P[i] will hold the Manacher‐style radius at center i
        P = [0] * length

        # inversion map for the 0/1 bits and the separator '#'
        inv = {'0':'1', '1':'0', '#':'#'}

        pos = 1   # center of the rightmost-reaching antisymmetry
        mx  = 1   # its right boundary = pos + P[pos]
        ans = 0

        # only odd i (the '#' positions) correspond to even‐length substrings
        for i in range(1, tot+1, 2):
            if i < mx:
                mirror = 2*pos - i
                # same as: len[i] = min(mx - i, len[mirror])
                P[i] = min(mx - i, P[mirror])
            else:
                P[i] = 1

            # expand as long as T[i + P] == inv[T[i - P]]
            while True:
                left = i - P[i]
                right = i + P[i]
                # boundary guard
                if left < 0 or right >= length:
                    break
                # must both be in our inv‐map (i.e. '#','0','1')
                cL = T[left]
                cR = T[right]
                if cL not in inv or cR not in inv:
                    break
                if cR == inv[cL]:
                    P[i] += 1
                else:
                    break

            # update the farthest-reaching center
            if i + P[i] > mx:
                mx  = i + P[i]
                pos = i

            # each full two‐step in the radius == one antisymmetric substring
            ans += (P[i] >> 1)

        self.parameter["reference_answer"] = ans
    

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
            if processed_result < 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                if self.parameter["reference_answer"] == 0 :
                    return self.rewards["rewarding_weight"] * int(processed_result == 0)
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]