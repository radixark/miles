import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class StringPartitionShuffle_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3785
    prompt_template = \
r"""You are given a string S of length {N} (0-indexed): {S}

Please find {K} intervals [L[1], R[1]), ..., [L[{K}], R[{K}]) such that:
- Each interval [L[i], R[i]) is non-empty and disjoint.
- The intervals together cover the entire string S (each index appears in exactly one interval).
- Concatenating all substrings S[L[i]: R[i]] (= S[L[i]] + S[L[i] + 1] + ... + S[R[i] - 1]) (in order) yields a new string T: {T}

**Output Format:** Output {K} lines. The i-th line should contain two integers L[i] and R[i], separated by a space."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([a=b])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the StringPartitionShuffle_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        if N >= 4 and random.random() < 0.5 :
            K = self.parameter["K"] = 3
        else :
            K = self.parameter["K"] = random.randint(2, N - 1)

        one_probability = random.uniform(0.1, 0.9)
        S = self.parameter["S"] = "".join("1" if random.random() < one_probability else "0" for _ in range(N))

        endpoints = random.sample(range(1, N), K - 1)
        endpoints.sort()
        endpoints = [0] + endpoints + [N]
        assert len(endpoints) == K + 1, "endpoints should have length K + 1"
        intervals = [(endpoints[i], endpoints[i + 1]) for i in range(K)]
        assert len(intervals) == K, "intervals should have length K"
        random.shuffle(intervals)
        self.parameter["T"] = "".join(S[L : R] for L, R in intervals)
        self.parameter["reference_answer"] = "\n".join("{} {}".format(L, R) for L, R in intervals)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            K = self.parameter["K"],
            S = self.parameter["S"],
            T = self.parameter["T"],
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                matrix = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        matrix.append(list(map(int, line.split())))
                return matrix
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            if len(processed_result) != self.parameter["K"] :
                return self.rewards["wrong_format"]
            if not all(len(interval) == 2 for interval in processed_result) :
                return self.rewards["wrong_format"]
            
            if not all(0 <= L < R <= self.parameter["N"] for L, R in processed_result) :
                return self.rewards["invalid_solution"]
            if not sum(R - L for L, R in processed_result) == self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not set(i for L, R in processed_result for i in range(L, R)) == set(range(self.parameter["N"])) :
                return self.rewards["invalid_solution"]
            
            T = "".join(self.parameter["S"][L : R] for L, R in processed_result)
            assert len(T) == self.parameter["N"] == len(self.parameter["T"]), "Length of T should match N"

            if self.rewards["rewarding_strategy"] == "mean([a=b])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["T"], T)) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "a=b" :
                return self.rewards["rewarding_weight"] * (self.parameter["T"] == T)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]