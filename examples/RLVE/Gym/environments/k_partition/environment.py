import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class KPartition_Environment(VerifiableEnvironment) : # Source : https://en.wikipedia.org/wiki/3-partition_problem
    prompt_template = \
r"""You are given a **multiset S** containing **{N}** positive integers: {Multiset_S}.
Given K=**{K}**, the **target value T** is calculated as the total sum of elements in **S**, divided by **{N} / K = {N} / {K} = {N_divided_by_K}**. 
Your task is to find a partition that divides **S** into **{N_divided_by_K}** disjoint **K-tuples** (S_1, S_2, ..., S_{K}), where these tuples **cover the entire set S**, and the sum of the elements in each **K-tuple** equals **T**.

**Output Format:** Your final answer should contain **{N_divided_by_K} lines**, each containing **{K}** integers representing a valid K-tuple from the partition (with elements separated by spaces)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the KPartition_Environment instance.
        """
        super().__init__(**kwargs)
        
        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }


    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 4, "N should be greater than or equal to 4"

        assert "K" in self.parameter, "K is required in parameter"
        K = self.parameter["K"]
        assert K >= 2, "K should be greater than or equal to 2"
        assert N % K == 0, "K should be a factor of N"

        T = self.parameter["T"] = random.randint(max(K, N * K // 10), N * K) # This can be adjusted
        
        # Generate N // K K-tuples, each summing to T
        N_divided_by_K = N // K
        Multiset_S = []
        tuples = []
        for _ in range(N_divided_by_K) :
            # Generate K - 1 random positive integers less than T
            cuts = sorted(random.sample(range(1, T), K - 1))
            tuple_vals = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, K - 1)] + [T - cuts[-1]]
            random.shuffle(tuple_vals)
            tuples.append(tuple_vals)
            Multiset_S.extend(tuple_vals)
        random.shuffle(Multiset_S)
        self.parameter["Multiset_S"] = Multiset_S
        self.parameter["reference_answer"] = "\n".join([" ".join(map(str, t)) for t in tuples])

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        K = self.parameter["K"]
        N_divided_by_K = self.parameter["N"] // self.parameter["K"]
        Multiset_S = self.parameter["Multiset_S"]
        assert len(Multiset_S) == N, "N should be the size of the multiset S"
        assert sum(Multiset_S) % N_divided_by_K == 0, "The sum of the multiset S should be a multiple of N/K"
        return self.prompt_template.format(
            N = N,
            K = K,
            N_divided_by_K = N_divided_by_K,
            Multiset_S = ", ".join(map(str, Multiset_S)),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                tuples = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        tuples.append(list(map(int, line.split())))
                return tuples
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            tuples = processed_result
            if len(tuples) != self.parameter["N"] // self.parameter["K"] :
                return self.rewards["invalid_solution"]
            
            for t in tuples : 
                if len(t) != self.parameter["K"] :
                    return self.rewards["invalid_solution"]
            
            # Flatten the tuples and compare with Multiset_S
            flat_output = sorted([item for group in tuples for item in group])
            multiset_s_sorted = sorted(self.parameter["Multiset_S"])
            assert len(flat_output) == len(multiset_s_sorted), "Flat output and multiset S should have the same length"
            if flat_output != multiset_s_sorted :
                return self.rewards["invalid_solution"]
            
            for t in tuples :
                if sum(t) != self.parameter["T"] :
                    return self.rewards["wrong_answer"]

            return self.rewards["correct_answer"]
        else :
            return self.rewards["wrong_format"]