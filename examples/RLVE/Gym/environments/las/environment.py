import random
from collections import deque
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class LAS_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3584
    prompt_template = \
r"""There are {N} people labeled from 1 to {N}, and {N} foods also labeled from 1 to {N}. The i-th food has C[i] calories, and the array C is: {C}

Each person chooses one food as follows:
- Person i (1 ≤ i < {N}) can choose either food i or food i+1.
- Person {N} can choose either food {N} or food 1.
- If a food is chosen by only one person, that person receives all of its calories. If a food is chosen by two people, they share the calories of that food **equally**.

You are to find a valid food assignment (i.e., choose one food between the two choices for each person), such that for **every person**, if this person switches to the other food choice (while all other people keep their choices unchanged), this person does **NOT** receive more calories than this person currently does.
**Output Format:** Output a single line with {N} integers — the food chosen by person 1, 2, ..., {N}, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the LAS_Environment instance.
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
        assert N >= 3, "N must be at least 3"

        A = self.parameter["A"] = [random.randint(1, 2 * N) for _ in range(N)]


        # B will hold the circular “Num” array (1‑indexed, with B[N+1] = B[1])
        B = [0] * (N + 2)
        for i in range(1, N + 1):
            B[i] = A[i - 1]
        B[N + 1] = B[1]

        # C is our DP table: (N+2) × 5, initialized to 0
        C = [[0] * 5 for _ in range(N + 2)]

        def Dynamic_Programming(s):
            # reset
            for i in range(N + 2):
                for j in range(5):
                    C[i][j] = 0
            # base case: at position 1, state s is reachable from “1”
            C[1][s] = 1

            # build DP up through i = N+1
            for i in range(2, N + 2):
                if C[i - 1][1] and B[i - 1] <= B[i] * 2:
                    C[i][1] = 1
                if C[i - 1][1] and B[i - 1] <= B[i]:
                    C[i][3] = 1
                if C[i - 1][2] and B[i] <= B[i - 1] * 2:
                    C[i][2] = 2
                if C[i - 1][2] and B[i] <= B[i - 1]:
                    C[i][4] = 2
                if C[i - 1][3] and B[i] <= B[i - 1]:
                    C[i][2] = 3
                if C[i - 1][3] and B[i] * 2 <= B[i - 1]:
                    C[i][4] = 3
                if C[i - 1][4] and B[i - 1] <= B[i]:
                    C[i][1] = 4
                if C[i - 1][4] and B[i - 1] * 2 <= B[i]:
                    C[i][3] = 4

            # return whether we can end in the same state s at position N+1
            return C[N + 1][s] != 0

        # D will store the final choices (1‑indexed)
        D = [0] * (N + 2)

        # Try all 4 possible end‑states
        for s in range(1, 5):
            if Dynamic_Programming(s):
                # reconstruct backwards
                x = s
                for j in range(N + 1, 0, -1):
                    if x == 1:
                        D[j - 1] = ((j - 1) % N) + 1
                    if x == 2:
                        D[j] = ((j - 1) % N) + 1
                    if x == 3:
                        D[j - 1] = ((j - 1) % N) + 1
                        D[j]     = ((j - 1) % N) + 1
                    # note: original C++ omitted an explicit case for x==4
                    x = C[j][x]

                # output persons 1..N
                self.parameter["reference_answer"] = " ".join(str(D[i]) for i in range(1, N + 1))
                break

    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            C = ", ".join("C[{}]={}".format(i + 1, Ci) for i, Ci in enumerate(self.parameter["A"])),
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
            choices = [choice - 1 for choice in processed_result]  # Convert to 0-based index

            if len(choices) != self.parameter["N"] :
                return self.rewards["wrong_format"]
            if not all(choice in (person, (person + 1) % self.parameter["N"]) for person, choice in enumerate(choices)) :
                return self.rewards["invalid_solution"]
            counting = [0] * self.parameter["N"]
            for choice in choices :
                counting[choice] += 1
            
            def get_calories(choice) :
                if counting[choice] == 1 :
                    return self.parameter["A"][choice] * 2
                elif counting[choice] == 2 :
                    return self.parameter["A"][choice] * 1
                else :
                    raise ValueError("Invalid counting for choice {}: {}".format(choice, counting[choice]))
            
            satisfied = 0
            for person, choice in enumerate(choices) :
                current = get_calories(choice)
                
                other_choice = ((person + (person + 1)) - choice) % self.parameter["N"]
                # counting[choice] -= 1
                counting[other_choice] += 1
                changed = get_calories(other_choice)
                # counting[choice] += 1
                counting[other_choice] -= 1

                satisfied += int(current >= changed)
            
            assert satisfied <= self.parameter["N"], "satisfied should not exceed N, got {}".format(satisfied)
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == self.parameter["N"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]