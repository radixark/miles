import random
from collections import deque
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class SlidingWindow_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1886
    prompt_template = \
r"""You are given the following list of {N} numbers: {A}
Please find the minimum value in each contiguous subarray of size {K} (there are {N_minus_K_plus_1} such subarrays in total).

Your final answer should be a single line containing the minimum values (from the leftmost subarray to the rightmost), separated by **spaces**."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the SlidingWindow_Environment instance.
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
        assert N >= 3, "N should be greater than or equal to 3"

        K = self.parameter["K"] = random.randint(2, N - 1)
        A = self.parameter["A"] = [random.randint(-(N // 20), +N) for _ in range(N)]


        min_deque = deque()  # will store indices, increasing by a[]
        self.parameter["gold_answer"] = mins = []

        for i in range(N) :
            if min_deque and min_deque[0] <= i - K :
                min_deque.popleft()
            while min_deque and A[min_deque[-1]] > A[i] :
                min_deque.pop()
            min_deque.append(i)
            if i >= K - 1 :
                mins.append(A[min_deque[0]])
        
        assert len(mins) == N - K + 1, "The length of gold_answer should be N - K + 1"
        self.parameter["reference_answer"] = " ".join(map(str, mins))
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        K = self.parameter["K"]
        return self.prompt_template.format(
            N = N,
            K = K,
            N_minus_K_plus_1 = N - K + 1,
            A = " ".join(map(str, self.parameter["A"])),
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

            if len(processed_result) != self.parameter["N"] - self.parameter["K"] + 1 :
                return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["gold_answer"], processed_result)) / (self.parameter["N"] - self.parameter["K"] + 1)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]