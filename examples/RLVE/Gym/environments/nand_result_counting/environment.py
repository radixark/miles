import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class NANDResultCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3220
    prompt_template = \
r"""From now on, all numbers are treated as {K}-bit binary strings (i.e., only the lowest {K} bits are considered, and leading zeros may be added to fill up to {K} bits).

The **NAND** operation is defined as:
- 0 NAND 0 = 1
- 0 NAND 1 = 1 NAND 0 = 1
- 1 NAND 1 = 0

You are given the following {N} numbers: {numbers}
You may combine them arbitrarily using the NAND operation and brackets (i.e., in any order, any number of times).

How many distinct numbers in the range [{L}, {R}] (inclusive) can be obtained by such combinations? Note: all intermediate and final results are considered as {K}-bit binary strings, so only numbers within the {K}-bit range are valid."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the NANDResultCounting_Environment instance.
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
        assert N >= 2, "N should be greater than or equal to 2"

        assert "K" in self.parameter, "K is required in parameter"
        K = self.parameter["K"]
        assert K >= 1, "K should be greater than or equal to 1"

        component_num = random.randint(1, K)
        endpoints = random.sample(range(1, K), component_num - 1) if component_num > 1 else []
        endpoints.sort()
        endpoints = [0] + endpoints + [K]
        assert len(endpoints) == component_num + 1, "Endpoints should be of length component_num + 1"
        allbits = list(range(K))
        random.shuffle(allbits)
        assert all(0 <= endpoints[i] < endpoints[i + 1] <= K for i in range(component_num)), "Endpoints should be in the range [0, K] and strictly increasing"
        components = [allbits[endpoints[i] : endpoints[i + 1]] for i in range(component_num)]

        def generate_number() -> int :
            number = 0
            existence_probability = random.random()
            for component in components :
                if random.random() < existence_probability :
                    number |= sum(1 << bit for bit in component)
            return number
        self.parameter["A"] = A = [generate_number() for _ in range(N)]

        L, R = random.randint(0, (1 << K) - 1), random.randint(0, (1 << K) - 1)
        if L > R:
            L, R = R, L
        self.parameter["L"], self.parameter["R"] = L, R


        full = (1 << K) - 1
        lk = [0] * K
        num = [0] * K
        have = 0

        # build the 'basis' masks
        for i in range(K - 1, -1, -1):
            if ((have >> i) & 1) == 0:
                now_mask = full
                for a in A:
                    if (a >> i) & 1:
                        now_mask &= a
                    else:
                        # mask off to K bits here!
                        now_mask &= (~a) & full
                lk[i] = now_mask
                num[i] = 1
                have |= now_mask

        # prefix‐sum the counts
        for i in range(1, K):
            num[i] += num[i - 1]

        def count_upto(x):
            # how many reachable values ≤ x
            if x < 0:
                return 0
            if x >= full:
                return 1 << num[K - 1]
            ans = 0
            for i in range(K - 1, -1, -1):
                if x < 0:
                    break
                if (x >> i) & 1:
                    if lk[i] != 0:
                        ans += 1 << (num[i] - 1)
                        x -= lk[i]
                    else:
                        ans += 1 << num[i]
                        break
            if x == 0:
                ans += 1
            return ans

        self.parameter["reference_answer"] = count_upto(R) - count_upto(L - 1)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            K = self.parameter["K"],
            numbers = " ".join(map(str, self.parameter["A"])),
            L = self.parameter["L"], R = self.parameter["R"],
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