import random
from typing import Optional, Tuple
from Gym.environment import VerifiableEnvironment


class SaladBar_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3564
    prompt_template = \
r"""You are given a string S (0-indexed) of length {N}, consisting only of the characters `j` and `p`: {S}

Please find a **contiguous** substring S[l : r] (using Python-style slicing: 0 ≤ l < r ≤ {N}, which includes S[l] through S[r - 1], but **NOT** S[r]) such that:
- In **every prefix** of the substring, the number of `p` characters is **not less than** the number of `j` characters.
- In **every suffix** of the substring, the number of `p` characters is **not less than** the number of `j` characters.

Your goal is to **maximize the length** of such a substring (i.e., maximize r - l). Output two integers `l` and `r`, separated by a space."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the SaladBar_Environment instance.
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

        p_probability = random.uniform(0.0, 0.7)
        while True :
            S = self.parameter["S"] = "".join("p" if random.random() < p_probability else "j" for _ in range(N))
            if "p" in S and "j" in S :
                break
        

        # Compute prefix sums and track minimum and maximum
        prefix = [0] * (N + 1)
        minx = 0
        maxx = 0
        for i in range(1, N + 1):
            prefix[i] = prefix[i - 1] + (1 if S[i - 1] == 'p' else -1)
            if prefix[i] < minx:
                minx = prefix[i]
            if prefix[i] > maxx:
                maxx = prefix[i]

        # Prepare linked lists for each adjusted prefix-sum value
        range_x = maxx - minx + 1
        head = [-1] * range_x
        nxt = [-1] * (N + 1)
        to = [0] * (N + 1)

        # Build next pointers for equal adjusted-sum indices
        for i in range(N, -1, -1):
            x = prefix[i] - minx
            nxt[i] = head[x]
            head[x] = i
            to[i] = i

        # Scan backwards to find longest valid segment
        ans = 0
        pre = N
        for i in range(N, 0, -1):
            if S[i - 1] == 'j':
                # Can't start with an apple
                pre = i - 1
            else:
                idx = i - 1
                ni = nxt[idx]
                # Potentially update end based on next equal-sum position
                if ni >= 0 and prefix[to[ni]] >= prefix[pre]:
                    pre = to[ni]
                to[idx] = pre
                length = pre - i + 1
                if length > ans:
                    self.parameter["reference_answer"] = "{} {}".format(i - 1, pre)
                    ans = length
        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], S = self.parameter["S"])


    def _process(self, answer : Optional[str]) -> Optional[Tuple[int, int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                l, r = map(int, answer.split())
                return l, r
            except :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            l, r = processed_result
            if not (0 <= l < r <= self.parameter["N"]) :
                return self.rewards["invalid_solution"]
            
            T = self.parameter["S"][l : r]
            def check(s) :
                counting = 0
                for c in s :
                    counting += (+1 if c == 'p' else -1)
                    if counting < 0 :
                        return False
                return True
            if not (check(T) and check(T[::-1])) :
                return self.rewards["invalid_solution"]
            
            gold, answer = self.parameter["gold_answer"], r - l
            assert 0 < answer <= gold, "answer should be less than or equal to gold"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]