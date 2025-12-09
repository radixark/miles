import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class ABProgramSimulation_Environment(VerifiableEnvironment) : # Source : https://x.com/VictorTaelin/status/1776096481704804789
    prompt_template = \
r"""A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`.

An A::B program is a sequence of tokens, e.g., `B# A# #B #A B#`.

To *compute* a program, we must rewrite neighbor tokens, using the rules (whenever two neighbor tokens have their `#` facing each-other, they must be rewritten according to the corresponding rule) :
+ `A# #A` ... becomes ... `` (nothing)
+ `A# #B` ... becomes ... `#B A#`
+ `B# #A` ... becomes ... `#A B#`
+ `B# #B` ... becomes ...  `` (nothing)

Please give the final state of the program: {program}
An example for output format: `B# A# A#`
"""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the AB_Program_Simulation_Environment instance.
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
        assert N >= 1, "N should be greater than or equal to 1"

        assert "max_steps" in self.parameter, "max_steps is required in parameter"
        max_steps = self.parameter["max_steps"]
        assert max_steps >= 1, "max_steps should be greater than or equal to 1"

        while True :
            distribution = [random.randint(1, N) for _ in range(4)]
            distribution = [d / sum(distribution) for d in distribution]
            self.parameter["program"] = [["A#", "#A", "B#", "#B"][i] for i in random.choices(range(4), distribution, k = N)]

            current, final = self.parameter["program"].copy(), None
            for step in range(max_steps) :
                new_program = None

                for i in range(len(current) - 1) :
                    a, b = current[i], current[i + 1]
                    if a == "A#" and b == "#A" :
                        new_program = current[: i] + current[i + 2 :]
                    elif a == "A#" and b == "#B" :
                        new_program = current[: i] + ["#B", "A#"] + current[i + 2 :]
                    elif a == "B#" and b == "#A" :
                        new_program = current[: i] + ["#A", "B#"] + current[i + 2 :]
                    elif a == "B#" and b == "#B" :
                        new_program = current[: i] + current[i + 2 :]
                    if new_program is not None:
                        break

                if new_program is None :
                    final = current
                    break
                else :
                    current = new_program
            
            if final is not None :
                self.parameter["reference_answer"] = " ".join(final)
                self.parameter["gold_answer"] = final
                break
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(program = " ".join(self.parameter["program"]))
    

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = answer.split()
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            if not all(token in ("A#", "#A", "B#", "#B") for token in processed_result) :
                return self.rewards["wrong_format"]
            
            if processed_result == self.parameter["gold_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]