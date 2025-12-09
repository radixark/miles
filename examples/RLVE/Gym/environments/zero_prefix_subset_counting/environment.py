import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class ZeroPrefixSubsetCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1666
    prompt_template = \
r"""You are given {N} strings:
{strings}

How many **non-empty** subsets such that **no string is a prefix of another string** within the subset?"""
    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the ZeroPrefixSubsetCounting_Environment instance.
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

        while True :
            proportion_being_prefix = random.uniform(0.1, 0.9)
            M = N - int(N * proportion_being_prefix)
            if M < 1 :
                continue
            array = self.parameter["array"] = []
            for i in range(M) :
                while True :
                    length = random.randint(2, N)
                    s = "".join(random.choices("ab", k = length))
                    if s not in array :
                        array.append(s)
                        break
            for i in range(N - M) :
                prefix = random.choice(array[: M])
                array.append(prefix[: random.randint(1, len(prefix) - 1)])
            assert len(array) == N
            if len(array) == len(set(array)) :
                random.shuffle(array)
                break
        

        A = [''] + array.copy()
        A = [''] + sorted(A[1:])  # sort a[1..N]

        # f and dp sized dynamically by N
        f = [[False] * (N + 1) for _ in range(N + 1)]
        dp = [0] * (N + 1)

        def calc(i, j):
            # Ensure the shorter (or equal) string is at i
            if len(A[i]) > len(A[j]):
                i, j = j, i
            # Return true iff A[i] is NOT a prefix of A[j]
            return A[j].find(A[i]) != 0

        for i in range(1, N + 1):
            dp[i] = 1
            for j in range(1, N + 1):
                f[i][j] = calc(i, j)

        for i in range(1, N + 1):
            for j in range(i, N + 1):
                if f[i][j]:
                    dp[j] += dp[i]

        ret = sum(dp[1:])
        self.parameter["reference_answer"] = ret
    
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            strings = "\n".join("String {}: {}".format(i, Si) for i, Si in enumerate(self.parameter["array"], start = 1)),
        )


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
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]