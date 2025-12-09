import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class TaskArrangement_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2365
    prompt_template = \
r"""You are given {N} tasks, numbered from 1 to {N}. Each task i (1 <= i <= {N}) takes T[i] units of time to complete individually and has a cost coefficient F[i]. The values are given as:
{T_and_F}

You may divide these tasks (in order) into any number of **consecutive batches**. Let the total number of batches be k (k >= 1), and let end[1], end[2], ..., end[k] (1 <= end[1] < end[2] < ... < end[k] = {N}) denote the last task index in each batch.
- This means:
    + Batch 1 contains tasks 1 to end[1]
    + Batch 2 contains tasks end[1] + 1 to end[2]
    + ...
    + Batch k contains tasks end[k - 1] + 1 to end[k] (with end[k] = {N})

- Before starting each batch, the machine must spend an additional {S} units of startup time.
- The time to **complete** a batch is the sum of T[i] for all tasks in that batch.
- Therefore, the **total completion time** of each task in a batch is the sum of the batch's startup time ({S}) and the total time of all tasks in that batch.
- All tasks in a batch are considered to finish **simultaneously**, at the end of that batch.

- Tasks are completed in the order defined by the batch division.
- The cost of each task is equal to **the time when its batch finishes (after all previous batches, if any, have completed and the current batch has been processed), multiplied by F[i]**.
- The **total cost** is the sum of the costs of all tasks.

Try your best to find a batch division (end[1], end[2], ..., end[k]) that **minimizes the total cost**.

**Output Format:**
Your final answer should be a single line containing end[1], end[2], ..., end[k] (with end[k] always equal to {N}), separated by **spaces**.
Example: `1 2 {N}` (do **NOT** include the backticks or quotes); this means:
- There are 3 batches,
- The first batch ends at task 1,
- The second batch ends at task 2,
- The last batch ends at task {N} and includes the remaining tasks.
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = +3.0,
                 **kwargs) :
        """
        Initialize the TaskArrangement_Environment instance.
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
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        S = self.parameter["S"] = random.randint(0, N * 3)
        T, F = [None] + [random.randint(1, N) for _ in range(N)], [None] + [random.randint(1, N) for _ in range(N)]
        self.parameter["T"], self.parameter["F"] = T[1 :], F[1 :]
        assert len(self.parameter["T"]) == N, "T should have length N"
        assert len(self.parameter["F"]) == N, "F should have length N"


        prefix_T = [0] * (N + 1)
        for i in range(1, N + 1) :
            prefix_T[i] = prefix_T[i - 1] + T[i]
        def sum_T(l, r) :
            return prefix_T[r] - prefix_T[l - 1]

        suffix_F = [0] * (N + 2)
        suffix_F[N + 1] = 0
        for i in range(N, 0, -1) :
            suffix_F[i] = suffix_F[i + 1] + F[i]

        prefix_F = [0] * (N + 1)
        for i in range(1, N + 1) :
            prefix_F[i] = prefix_F[i - 1] + F[i]
        def sum_F(l, r) :
            return prefix_F[r] - prefix_F[l - 1]

        dpF, dpG = [None] * (N + 1), [None] * (N + 1)
        dpF[0] = 0
        for i in range(1, N + 1) :
            for j in range(1, i + 1) :
                val = dpF[j - 1] + (S + sum_T(j, i)) * suffix_F[j]
                if dpF[i] is None or dpF[i] > val :
                    dpF[i] = val
                    dpG[i] = j

        ends = []
        now = N
        while now :
            ends.append(now)
            now = dpG[now] - 1
        ends.reverse()

        answer, current_time, last = 0, 0, 0
        for end in ends :
            current_time += S + sum_T(last + 1, end)
            answer += current_time * sum_F(last + 1, end)
            last = end
        assert answer == dpF[N]
        
        self.parameter["reference_answer"] = " ".join(map(str, ends))
        self.parameter["reference_answer_cost"] = answer
        assert answer > 0, "answer should be greater than 0"
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            S = self.parameter["S"],
            T_and_F = "\n".join("T[{}]={} F[{}]={}".format(i, self.parameter["T"][i - 1], i, self.parameter["F"][i - 1]) for i in range(1, N + 1)),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                if not answer_array :
                    return None
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            N = self.parameter["N"]

            ends = processed_result
            for i in range(len(ends)) :
                if not (1 <= ends[i] <= N) :
                    return self.rewards["invalid_solution"]
                if i and not (ends[i - 1] < ends[i]) :
                    return self.rewards["invalid_solution"]
            if ends[-1] != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            
            T, F = [None] + self.parameter["T"], [None] + self.parameter["F"]

            prefix_T = [0] * (N + 1)
            for i in range(1, N + 1) :
                prefix_T[i] = prefix_T[i - 1] + T[i]
            def sum_T(l, r) :
                return prefix_T[r] - prefix_T[l - 1]

            suffix_F = [0] * (N + 2)
            suffix_F[N + 1] = 0
            for i in range(N, 0, -1) :
                suffix_F[i] = suffix_F[i + 1] + F[i]

            prefix_F = [0] * (N + 1)
            for i in range(1, N + 1) :
                prefix_F[i] = prefix_F[i - 1] + F[i]
            def sum_F(l, r) :
                return prefix_F[r] - prefix_F[l - 1]
            
            answer, current_time, last = 0, 0, 0
            for end in ends :
                current_time += self.parameter["S"] + sum_T(last + 1, end)
                answer += current_time * sum_F(last + 1, end)
                last = end
            gold = self.parameter["reference_answer_cost"]
            assert gold <= answer, "answer should be greater than or equal to gold"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]