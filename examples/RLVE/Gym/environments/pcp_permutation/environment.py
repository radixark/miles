import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class PCPPermutation_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given two arrays of strings, `A` and `B`, each containing {N} strings:
{A_and_B}

Find a permutation p_0, ..., p_{N_minus_1} of the indices `0` to `{N_minus_1}` such that: `A[p_0] + ... + A[p_{N_minus_1}]` is equal to `B[p_0] + ... + B[p_{N_minus_1}]` (here, `+` denotes string concatenation).

**Output Format:** Your final answer should be a single line containing the permutation `p_0 ... p_{N_minus_1}`, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([a=b])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the PCPPermutation_Environment instance.
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
        assert N >= 2, "N should be greater than or equal to 2"

        assert "average_length" in self.parameter, "average_length is required in parameter"
        average_length = self.parameter["average_length"]
        assert average_length >= 1.0, "average_length should be greater than or equal to 1.0"

        sum_length = max(N + 1, random.randint(N, int(N * average_length)))
        probability = random.random()
        S = "".join("ab"[random.random() < probability] for _ in range(sum_length))

        for array_name in ("A", "B") :
            endpoints = random.sample(range(1, sum_length), N - 1)
            endpoints.sort()
            endpoints = [0] + endpoints + [sum_length]
            assert len(endpoints) == N + 1, "endpoints should have length N + 1"
            self.parameter[array_name] = [S[endpoints[i] : endpoints[i + 1]] for i in range(N)]

        permutation = list(range(N))
        random.shuffle(permutation)
        for array_name in ("A", "B") :
            self.parameter[array_name] = [self.parameter[array_name][i] for i in permutation]
        
        inv_permutation = [None] * N
        for i, p in enumerate(permutation) :
            inv_permutation[p] = i
        self.parameter["reference_answer"] = " ".join(map(str, inv_permutation))
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            A_and_B = "\n".join("A[{}]={} B[{}]={}".format(i, self.parameter["A"][i], i, self.parameter["B"][i]) for i in range(N)),
        )


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

            permutation = processed_result
            if len(permutation) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if len(set(permutation)) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= i < self.parameter["N"] for i in permutation) :
                return self.rewards["invalid_solution"]

            concatenated_A = "".join(self.parameter["A"][i] for i in permutation)
            concatenated_B = "".join(self.parameter["B"][i] for i in permutation)
            assert len(concatenated_A) == len(concatenated_B), "concatenated_A and concatenated_B should have the same length"
            if self.rewards["rewarding_strategy"] == "mean([a=b])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(concatenated_A, concatenated_B)) / len(concatenated_A)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "a=b" :
                return self.rewards["rewarding_weight"] * (concatenated_A == concatenated_B)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]