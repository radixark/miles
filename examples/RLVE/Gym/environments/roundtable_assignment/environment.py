import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class RoundTableAssignment_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""There are {M} groups of people and {N} tables.
- The i-th group consists of R[i] people. Array R: {R}
- The j-th table can seat up to C[j] people. Array C: {C}

You need to assign each person to a table such that:
- No table contains more than one person from the same group.
- No table exceeds its total capacity.

**Output Format:** Output {M} lines. The i-th line (0-indexed) should contain R[i] integers (separated by spaces), representing the table indices assigned to each person in the i-th group."""

    def __init__(self,
                 wrong_format: float = -1.0, invalid_solution: float = -0.5, rewarding_strategy: str = "(satisfied/all)^beta", rewarding_weight: float = +1.0, rewarding_beta: float = 5.0,
                 **kwargs) :
        """
        Initialize the RoundTableAssignment_Environment instance.
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
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 3, "MAX_N_M should be greater than or equal to 3"

        M = self.parameter["M"] = random.randint(2, MAX_N_M)
        R = self.parameter["R"] = []
        tables = [[] for table_index in range(MAX_N_M)]
        for group_index in range(M) :
            R.append(random.randint(2, MAX_N_M))
            table_indices = random.sample(range(MAX_N_M), R[-1])
            for table_index in table_indices :
                tables[table_index].append(group_index)
        tables = [table for table in tables if len(table) > 0]
        assert len(R) == M, "R should have length M"
        
        self.parameter["N"] = len(tables)
        self.parameter["C"] = [len(table) for table in tables]
        assert len(self.parameter["C"]) == self.parameter["N"], "C should have length N"

        reference_answer = [[] for group_index in range(M)]
        for table_index, table in enumerate(tables) :
            for group_index in table :
                reference_answer[group_index].append(table_index)
        assert all(len(answer) == R[group_index] for group_index, answer in enumerate(reference_answer)), "Reference answer does not match the group sizes"
        self.parameter["reference_answer"] = "\n".join(" ".join(map(str, answer)) for answer in reference_answer)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            M = self.parameter["M"],
            N = self.parameter["N"],
            R = " ".join("R[{}]={}".format(i, Ri) for i, Ri in enumerate(self.parameter["R"])),
            C = " ".join("C[{}]={}".format(i, Ci) for i, Ci in enumerate(self.parameter["C"])),
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

            if len(processed_result) != self.parameter["M"] :
                return self.rewards["invalid_solution"]
            
            countings = [0] * self.parameter["N"]
            for answer, Ri in zip(processed_result, self.parameter["R"]) :
                if len(answer) != Ri :
                    return self.rewards["invalid_solution"]
                if not all(0 <= i < self.parameter["N"] for i in answer) :
                    return self.rewards["invalid_solution"]
                if len(set(answer)) != Ri :
                    return self.rewards["invalid_solution"]
                for table_index in answer :
                    countings[table_index] += 1

            assert len(countings) == len(self.parameter["C"]) == self.parameter["N"], "countings should match the number of tables"
            satisfied = sum(int(counting <= Ci) for counting, Ci in zip(countings, self.parameter["C"]))
            assert satisfied <= self.parameter["N"], "satisfied should not exceed N"
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == self.parameter["N"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]