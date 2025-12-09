import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class HungryRabbit_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3895
    prompt_template = \
r"""Let's construct {M} sets of integers S(1), S(2), ..., S(M), where each set contains exactly {K} integers chosen from 1 to {N}. The following conditions must hold:
- For all i (2 ≤ i ≤ {M}), we have {K} - |S(i) ∩ S(i - 1)| ≤ {L}.
{constraints}

Output {M} lines, where the i-th line contains the {K} integers (in the range of [1, {N}]) in S(i), separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the HungryRabbit_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "unsuccessful_solution": unsuccessful_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 4, "MAX_N_M should be greater than or equal to 4"

        N = self.parameter["N"] = random.randint(4, MAX_N_M)
        M = self.parameter["M"] = random.randint(3, MAX_N_M)
        K = self.parameter["K"] = random.randint(2, N - 2)
        L = self.parameter["L"] = random.randint(1, K - 1)


        self.parameter["reference_answer"] = []
        forbidden = self.parameter["forbidden"] = []
        for i in range(M) :
            if i == 0 :
                S_i = random.sample(range(1, N + 1), k = K)
            else :
                S_i_minus_1 = self.parameter["reference_answer"][-1]
                S_i_minus_1_complement = list(set(range(1, N + 1)) - set(S_i_minus_1))
                num_diff = random.randint(0, min((L, len(S_i_minus_1), len(S_i_minus_1_complement))))
                S_i = random.sample(S_i_minus_1, k = K - num_diff) + random.sample(S_i_minus_1_complement, k = num_diff)
            random.shuffle(S_i)
            assert len(S_i) == K, "Length of S(i) must be K"
            self.parameter["reference_answer"].append(S_i)
            S_i_complement = list(set(range(1, N + 1)) - set(S_i))
            forbidden.append(sorted(random.sample(S_i_complement, k = random.randint(1, len(S_i_complement)))))
        self.parameter["reference_answer"] = "\n".join(" ".join(map(str, S_i)) for S_i in self.parameter["reference_answer"])
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            K = self.parameter["K"],
            L = self.parameter["L"],
            constraints = "\n".join("- S({}) must not contain any of the forbidden integers: {}".format(i + 1, " ".join(map(str, forbidden_i))) for i, forbidden_i in enumerate(self.parameter["forbidden"])),
        )


    def _process(self, answer : Optional[str]) -> Optional[List[List[int]]] :
        if answer is not None :
            answer = answer.strip()
            try :
                Sets = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        Sets.append(list(map(int, line.split())))
                return Sets
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            Sets = processed_result
            if len(Sets) != self.parameter["M"] :
                return self.rewards["invalid_solution"]
            if not all(len(Set) == self.parameter["K"] and len(set(Set)) == self.parameter["K"] for Set in Sets) :
                return self.rewards["invalid_solution"]
            if not all(1 <= x <= self.parameter["N"] for Set in Sets for x in Set) :
                return self.rewards["invalid_solution"]
            
            if not all(not (set(Set_i) & set(forbidden_i)) for Set_i, forbidden_i in zip(Sets, self.parameter["forbidden"])) :
                return self.rewards["unsuccessful_solution"]
            
            satisfied = sum(int(self.parameter["K"] - len(set(Sets[i]) & set(Sets[i - 1])) <= self.parameter["L"]) for i in range(1, self.parameter["M"]))
            assert 0 <= satisfied <= self.parameter["M"] - 1, "satisfied should be between 0 and M-1"
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / (self.parameter["M"] - 1)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == (self.parameter["M"] - 1))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]