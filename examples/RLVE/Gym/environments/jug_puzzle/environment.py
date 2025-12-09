import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class JugPuzzle_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given {N} jugs (initially empty) with the following capacities:
{jug_capacities}

Please fill a jug (you pick the one) with exactly {target_volumn} liters of water. You may perform the following actions:
- `Fill i` — Fill jug `i` to its full capacity.
- `Empty i` — Empty all water from jug `i`.
- `Pour i j` — Pour water from jug `i` to jug `j` until either jug `i` is empty or jug `j` is full.

**Output Format:** Each action should be written on its own line in the format shown above (without backticks or quotes). Output one action per line, in the order they should be performed."""

    def __init__(self,
                 max_capacity_multiple : int = 10,
                 operation_probabilities : Optional[List[float]] = [0.1, 0.1, 0.8],
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, wrong_solution : float = 0.0, correct_solution : float = 1.0,
                 **kwargs) :
        """
        Initialize the JugPuzzle_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_capacity_multiple = max_capacity_multiple

        assert len(operation_probabilities) == 3, "operation_probabilities should have exactly 3 elements for Fill, Empty, and Pour operations"
        assert sum(operation_probabilities) > 0, "operation_probabilities should sum to a positive value"
        self.operation_probabilities = operation_probabilities

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "wrong_solution" : wrong_solution,
            "correct_solution" : correct_solution,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        capacities = self.parameter["jug_capacities"] = [random.randint(2, N * self.max_capacity_multiple) for _ in range(N)]
        differences = set(capacity_i - capacity_j for capacity_j in capacities for capacity_i in capacities if capacity_i != capacity_j)

        jug = random.randint(0, N - 1)
        self.parameter["reference_answer"] = "Fill {}".format(jug)
        self.parameter["target_volumn"] = capacities[jug]

        assert "steps" in self.parameter, "steps is required in parameter"
        steps = self.parameter["steps"]
        assert steps >= 2, "steps should be greater than or equal to 2"

        volumns = [0] * N
        actions = ""
        existing_volumns = set()
        for step in range(steps) :
            while True :
                operation = random.choices(["Fill", "Empty", "Pour"], self.operation_probabilities)[0]
                if operation == "Fill" :
                    jug = random.randint(0, N - 1)
                    if volumns[jug] < capacities[jug] :
                        actions += "Fill {}\n".format(jug)
                        volumns[jug] = capacities[jug]
                        break
                elif operation == "Empty" :
                    jug = random.randint(0, N - 1)
                    if volumns[jug] > 0 :
                        actions += "Empty {}\n".format(jug)
                        volumns[jug] = 0
                        break
                elif operation == "Pour" :
                    jug_i = random.randint(0, N - 1)
                    jug_j = random.randint(0, N - 1)
                    if jug_i != jug_j and volumns[jug_i] > 0 and volumns[jug_j] < capacities[jug_j] :
                        actions += "Pour {} {}\n".format(jug_i, jug_j)
                        pour_amount = min(volumns[jug_i], capacities[jug_j] - volumns[jug_j])
                        volumns[jug_i] -= pour_amount
                        volumns[jug_j] += pour_amount
                        break
            
            target_volumns = set(volumn for volumn in volumns if volumn > 0) - existing_volumns - differences - set(capacities)
            if target_volumns :
                self.parameter["reference_answer"] = actions
                self.parameter["target_volumn"] = random.choice(list(target_volumns))
                existing_volumns |= target_volumns
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            target_volumn = self.parameter["target_volumn"],
            jug_capacities = "\n".join("Jug {}'s capacity: {} liters".format(i, capacity) for i, capacity in enumerate(self.parameter["jug_capacities"])),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            actions = []
            for line in answer.splitlines() :
                line = line.strip()
                if line :
                    actions.append(line.split())
                    action = actions[-1]
                    if not action :
                        return None
                    if action[0] in ("Fill", "Empty") :
                        if len(action) != 2 :
                            return None
                        try :
                            action[1] = int(action[1])
                        except ValueError :
                            return None
                    elif action[0] == "Pour" :
                        if len(action) != 3 :
                            return None
                        try :
                            action[1] = int(action[1])
                            action[2] = int(action[2])
                        except ValueError :
                            return None
                    else :
                        return None
            return actions
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            volumns = [0] * self.parameter["N"]
            for action in processed_result :
                if action[0] == "Fill" :
                    jug = action[1]
                    if not (0 <= jug < self.parameter["N"]) :
                        return self.rewards["invalid_solution"]
                    volumns[jug] = self.parameter["jug_capacities"][jug]
                elif action[0] == "Empty" :
                    jug = action[1]
                    if not (0 <= jug < self.parameter["N"]) :
                        return self.rewards["invalid_solution"]
                    volumns[jug] = 0
                elif action[0] == "Pour" :
                    jug_i, jug_j = action[1], action[2]
                    if not (0 <= jug_i < self.parameter["N"] and 0 <= jug_j < self.parameter["N"] and jug_i != jug_j) :
                        return self.rewards["invalid_solution"]
                    pour_amount = min(volumns[jug_i], self.parameter["jug_capacities"][jug_j] - volumns[jug_j])
                    volumns[jug_i] -= pour_amount
                    volumns[jug_j] += pour_amount
                else :
                    assert False, "Should be unreachable"
            
            if self.parameter["target_volumn"] in volumns :
                return self.rewards["correct_solution"]
            else :
                return self.rewards["wrong_solution"]
        else :
            return self.rewards["wrong_format"]