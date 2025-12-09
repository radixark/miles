import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class PrefixProductMODDistinctPermutation_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Please find a permutation of the numbers from 1 to {N} such that all {N} prefix products (i.e., the product of the first i numbers for all i from 1 to {N}) are **distinct modulo {N}**. Output the permutation as {N} integers (in order) in one line, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the PrefixProductMODDistinctPermutation_Environment instance.
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
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 3, "MAX_N should be greater than or equal to 3"

        while True :
            N = self.parameter["N"] = random.randint(3, MAX_N)

            def is_composite(x):
                """Return True if x is composite (has a non‑trivial divisor), False otherwise."""
                for i in range(2, int(x**0.5) + 1):
                    if x % i == 0:
                        return True
                return False

            if N == 1:
                assert False, "N should not be 1"
            elif N == 4:
                self.parameter["reference_answer"] = "1 3 2 4"
                break
            elif is_composite(N):
                continue
            else:
                # Compute modular inverses mod N in O(N)
                inv = [0] * (N + 1)
                inv[0] = inv[1] = 1
                for i in range(2, N + 1):
                    inv[i] = ((N - N//i) * inv[N % i]) % N
                # Build the sequence
                perm = [1]
                for i in range(1, N - 1):
                    perm.append(((i+1) * inv[i]) % N)
                perm.append(N)
                self.parameter["reference_answer"] = " ".join(map(str, perm))
                break


    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"])
    

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            if len(processed_result) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if set(processed_result) != set(range(1, self.parameter["N"] + 1)) :
                return self.rewards["invalid_solution"]
            
            existing, prefix_product = [False] * self.parameter["N"], 1
            for x in processed_result :
                prefix_product = (prefix_product * x) % self.parameter["N"]
                assert 0 <= prefix_product < self.parameter["N"], "prefix_product should be in the range [0, N)"
                existing[prefix_product] = True
            satisfied = sum(existing)
            assert 1 <= satisfied <= self.parameter["N"], "satisfied should be less than or equal to N"
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == self.parameter["N"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]