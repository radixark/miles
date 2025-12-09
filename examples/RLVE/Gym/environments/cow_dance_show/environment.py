import heapq
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class CowDanceShow_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3611
    prompt_template = \
r"""There are {N} cows labeled from 1 to {N}, and the i-th cow takes d[i] time to dance. The array d is given as: {d}

The cows dance on the stage as follows:
- Initially, the first {K} cows (cows 1 through {K}) are on the stage.
- Each cow dances for its own time d[i]. When a cow finishes dancing, it leaves the stage.
- As soon as a cow leaves, the next available cow in label order (if any) **immediately** takes its place. For example, when the first cow leaves, cow {K} + 1 enters the stage.

I am asking you to output the time when all cows have finished dancing."""
    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the CowDanceShow_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        self.parameter["d"] = [random.randint(1, N) for _ in range(N)]
        self.parameter["K"] = random.randint(2, N - 1)


        def compute(K):
            cow = self.parameter["d"].copy()
            # Initialize a min-heap with the first K cows
            heap = cow[:K]
            heapq.heapify(heap)
            # For each remaining cow, schedule it on the earliest free spot
            for i in range(K, N):
                t = heapq.heappop(heap)
                heapq.heappush(heap, t + cow[i])
            # The total time is the maximum finish time on stage
            return max(heap)
        self.parameter["reference_answer"] = compute(self.parameter["K"])
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            K = self.parameter["K"],
            d = ", ".join("d[{}]={}".format(i, di) for i, di in enumerate(self.parameter["d"], start = 1)),
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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]