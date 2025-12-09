import random
from typing import Optional, List, Tuple
from Gym.environment import VerifiableEnvironment


class DifferentColorPairing_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2902
    prompt_template = \
r"""There are {N} pearls, and each pearl has a color labeled from 1 to {M}. The number of pearls of each color is given as follows:
{C}

Please form exactly {N_div_2} pairs of pearls such that (1) each pearl belongs to exactly one pair; (2) the two pearls in each pair must have different colors. Output {N_div_2} lines, each containing two integers (separated by a space), representing the colors of the two pearls in one pair."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the DifferentColorPairing_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 6, "N should be greater than or equal to 6"
        assert N % 2 == 0, "N should be even"

        M = self.parameter["M"] = random.randint(3, N - 1)

        while True :
            C = random.sample(range(1, N), M - 1)
            C.sort()
            C += [N]
            for i in range(M - 1, 0, -1) :
                C[i] -= C[i - 1]
            assert len(C) == M
            assert sum(C) == N
            assert all(Ci > 0 for Ci in C)
            if not any(Ci > N - Ci for Ci in C) :
                self.parameter["C"] = C
                break
            
        # Expand colors: 1 repeated C[0] times, 2 repeated C[1] times, ...
        colors = []
        for idx, cnt in enumerate(C, start=1):
            if cnt > 0:
                colors.extend([idx] * cnt)

        # Output pairs: i with i + N//2
        half = N // 2
        self.parameter["reference_answer"] = ""
        for i in range(half):
            self.parameter["reference_answer"] += "{} {}\n".format(colors[i], colors[i + half])
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_div_2 = N // 2,
            M = self.parameter["M"],
            C = "\n".join("Color {} has {} pearls".format(i, Ci) for i, Ci in enumerate(self.parameter["C"], start = 1)),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List[Tuple[int, int]]] :
        if answer is not None :
            answer = answer.strip()
            try :
                pairs = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        c1, c2 = map(int, line.split())
                        pairs.append((c1, c2))
                return pairs
            except :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if len(processed_result) != self.parameter["N"] // 2 :
                return self.rewards["wrong_format"]
            if not all(1 <= c1 <= self.parameter["M"] and 1 <= c2 <= self.parameter["M"] and c1 != c2 for c1, c2 in processed_result) :
                return self.rewards["invalid_solution"]
            
            C = [0] * self.parameter["M"]
            for c1, c2 in processed_result :
                C[c1 - 1] += 1
                C[c2 - 1] += 1
            satisfied = sum(Ci == gold_Ci for Ci, gold_Ci in zip(C, self.parameter["C"]))
            assert satisfied <= self.parameter["M"], "Satisfaction level exceeded"
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / self.parameter["M"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == self.parameter["M"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]