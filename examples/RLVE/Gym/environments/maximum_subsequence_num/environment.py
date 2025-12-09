import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Maximum_SubsequenceNum_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""We want to obtain a sequence of length {M} + {N} = {M_plus_N} from an initial sequence of length {M} by appending {N} integers, each in [0, {K}). The initial sequence of length {M}: {A_first_M}

Try your best to maximize the number of essentially different subsequences of the final sequence.
Subsequence: picking some (>= 1) integers from the sequence in order, not necessarily contiguous.
Essentially different: only the sequence of values matters — same values in the same relative order are considered the same.

Your final answer should be a single line containing the {N} integers you appended to the initial sequence, separated by spaces, each in [0, {K}).
"""
    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Maximum_SubsequenceNum_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def subsequence_num(self, A : List[int]) :
        M, N, K = self.parameter["M"], self.parameter["N"], self.parameter["K"]
        assert len(A) == M + N + 1
        F = [0] * (M + N + 1)
        F[0] = 1
        last = [0] * K
        for i in range(1, M + N + 1) :
            if last[A[i]] == 0 :
                F[i] = F[i - 1] * 2
            else :
                F[i] = F[i - 1] * 2 - F[last[A[i]] - 1]
            last[A[i]] = i
        return F[M + N] - 1

    def _generate(self) -> None :
        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert M >= 1, "M should be greater than or equal to 1"

        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 1, "N should be greater than or equal to 1"

        assert "K" in self.parameter, "K is required in parameter"
        K = self.parameter["K"]
        assert K >= 2, "K should be greater than or equal to 2"

        self.parameter["A"] = [random.randint(0, K - 1) for i in range(1, M + 1)]


        A = [-1] + self.parameter["A"]
        assert len(A) == M + 1

        last = [0] * K
        for i in range(1, M + 1) :
            last[A[i]] = i
        for i in range(M + 1, M + N + 1) :
            k = min(range(K), key = lambda k : last[k])
            A.append(k)
            last[k] = i
        self.parameter["reference_answer"] = " ".join(str(a) for a in A[M + 1 : M + N + 1])
        self.parameter["gold_answer"] = self.subsequence_num(A)
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            M = self.parameter["M"],
            N = self.parameter["N"],
            K = self.parameter["K"],
            M_plus_N = self.parameter["M"] + self.parameter["N"],
            A_first_M = " ".join(map(str, self.parameter["A"])),
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

            A = processed_result
            if len(A) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= a < self.parameter["K"] for a in A) :
                return self.rewards["invalid_solution"]
            A = [-1] + self.parameter["A"] + A
            
            answer, gold = self.subsequence_num(A), self.parameter["gold_answer"]
            assert answer <= gold, "answer should be less than or equal to gold"
            
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]