import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class QuantumLockPuzzle_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""There is a 0/1 variable X, which is initially 0. You also have a variable Y, which starts at {Y_start}. You can press the buttons in any order, and you may press the same button multiple times. There are {N} buttons in total. Each time you press **any** button, X toggles: it becomes 1 - X.

When X is 0 and you press a button, Y changes according to the following rules:
{X0_rules}

When X is 1 and you press a button, Y changes according to the following rules:
{X1_rules}

Please find a sequence of button presses that will make Y equal to {Y_target}.

**Output Format:** Your final answer should be a single line containing the sequence of button presses in order, separated by spaces. For example, `0 1 0 2` means you pressed button 0, then button 1, then button 0 again, and finally button 2. Do **NOT** include backticks or quotes in your output."""

    def __init__(self,
                 operation_weights : Optional[List[float]] = [0.4, 0.4, 0.2],
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, wrong_solution : float = 0.0, correct_solution : float = 1.0,
                 **kwargs) :
        """
        Initialize the QuantumLockPuzzle_Environment instance.
        """
        super().__init__(**kwargs)

        self.operation_weights = operation_weights

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "wrong_solution" : wrong_solution,
            "correct_solution" : correct_solution,
        }
    

    def operate(self, Y : int, rule : List) -> int :
        operation, value = rule
        if operation == "+" :
            return Y + value
        elif operation == "-" :
            return Y - value
        elif operation == "*" :
            return Y * value
        else :
            raise NotImplementedError(f"Unknown operation: {operation}")
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        Y = self.parameter["Y_start"] = random.randint(-N, +N)
        buttons = self.parameter["buttons"] = []
        for button in range(N) :
            def rule_generator() :
                operation = random.choices(["+", "-", "*"], weights = self.operation_weights, k = 1)[0]
                if operation in ("+", "-") :
                    value = random.randint(1, N)
                elif operation in ("*", ) :
                    value = random.randint(2, 3)
                else :
                    raise NotImplementedError
                return [operation, value]
            buttons.append([rule_generator() for _ in range(2)])
        
        steps = self.parameter["steps"]
        assert steps >= 2, "steps should be greater than or equal to 2"
        steps += random.randint(0, 1)

        X = 0
        pressed_buttons = []
        existing_Y = set([Y])
        for step in range(steps) :
            button = random.randint(0, N - 1)
            pressed_buttons.append(button)
            Y = self.operate(Y, buttons[button][X])
            X = 1 - X
            if Y not in existing_Y :
                existing_Y.add(Y)
                self.parameter["reference_answer"] = pressed_buttons.copy()
                self.parameter["Y_target"] = Y
        if "Y_target" not in self.parameter :
            assert Y == self.parameter["Y_start"]
            self.parameter["reference_answer"] = ""
            self.parameter["Y_target"] = self.parameter["Y_start"]
        else :
            self.parameter["reference_answer"] = " ".join(map(str, self.parameter["reference_answer"]))
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            Y_start = self.parameter["Y_start"],
            Y_target = self.parameter["Y_target"],
            X0_rules = "\n".join("When you press button {}, Y becomes Y {} {}".format(i, button[0][0], button[0][1]) for i, button in enumerate(self.parameter["buttons"])),
            X1_rules = "\n".join("When you press button {}, Y becomes Y {} {}".format(i, button[1][0], button[1][1]) for i, button in enumerate(self.parameter["buttons"])),
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

            X, Y = 0, self.parameter["Y_start"]
            for button in processed_result :
                if not (0 <= button < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                Y = self.operate(Y, self.parameter["buttons"][button][X])
                X = 1 - X
            
            if Y == self.parameter["Y_target"] :
                return self.rewards["correct_solution"]
            else :
                return self.rewards["wrong_solution"]
        else :
            return self.rewards["wrong_format"]