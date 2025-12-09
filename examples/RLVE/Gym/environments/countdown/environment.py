import re
import math
import random
from abc import abstractmethod
from typing import Optional, List, Dict
from Gym.environment import VerifiableEnvironment


class Countdown_Environment(VerifiableEnvironment) :
    operations = ("+", "-", "*", "/")
    epsilon = 1E-5

    @abstractmethod
    def _check_parameter(self) -> bool :
        """
        Check if the parameter is valid.

        Returns:
            bool: True if the parameter is valid, False otherwise.
        """
        pass

    def _generate(self) -> None :
        assert "max_target" in self.parameter, "max_target is required in parameter"
        max_target = self.parameter["max_target"]
        assert max_target >= 0, "max_target should be greater than or equal to 0"

        assert "max_operand" in self.parameter, "max_operand is required in parameter"
        max_operand = self.parameter["max_operand"]
        assert max_operand >= 1, "max_operand should be greater than or equal to 1"

        assert "num_operands" in self.parameter, "num_operands is required in parameter"
        num_operands = self.parameter["num_operands"]
        assert num_operands >= 2, "num_operands should be greater than or equal to 2"

        while True :
            self.parameter["target"] = random.randint(0, max_target)
            self.parameter["operands"] = [random.randint(1, max_operand) for _ in range(num_operands)]
            assert len(self.parameter["operands"]) == num_operands, "Invalid number of operands"
            if self._check_parameter() :
                break
    

    def _prompt_generate(self) -> str :
        return self._prompt_template().format(target = self.parameter["target"], operands = " ".join(map(str, self.parameter["operands"])), operations = ", ".join(self.operations))
    
    @abstractmethod
    def _prompt_template(self) -> str :
        pass
    

    def _process(self, answer : Optional[str]) -> Dict :
        if answer is not None :
            answer = answer.strip()
            
            def calculate_expression() :
                allowed_pattern = r"^[\d+\-*/().\s]+$"
                if not re.match(allowed_pattern, answer) :
                    raise ValueError("Invalid characters in expression")
                res = eval(answer, {"__builtins__" : None}, {})
                try :
                    if not math.isfinite(float(res)) :
                        return None
                    return res
                except :
                    return None
            
            def valid_expression() -> bool :
                used_operands = sorted([int(operand) for operand in re.findall(r"\d+", answer)])
                available_operands = sorted(self.parameter["operands"])
                return used_operands == available_operands
            
            try :
                return {"format" : True, "result" : calculate_expression() if valid_expression() else None}
            except :
                return {"format" : False}
        else :
            return {"format" : False}


class CountdownEqual_Environment(Countdown_Environment) :
    def __init__(self,
                 wrong_format : float = -1.0, invalid_expression : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the CountdownEqual_Environment instance.

        Args:
            reward (dict): Dictionary of rewards for different evaluation results.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_expression" : invalid_expression,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }


    def _prompt_template(self) -> str :
        return \
        r"""You are given the following numbers: {operands}
Your task is to create a valid mathematical expression that evaluates to the target number: **{target}**.

Constraints:
- Use **each number exactly once**
- Only these basic arithmetic operations are allowed: {operations}
- You may use **parentheses** to define order of operations
- Your answer should be a **single valid expression**, with no extra explanation

Provide only the expression as your final answer."""


    def _check_parameter(self) -> bool :
        visited = set()
        def search(operands : List[int]) -> bool :
            if len(operands) == 1 :
                return operands[0] == self.parameter["target"]
            
            sorted_operands = tuple(sorted(operands))
            if sorted_operands in visited :
                return False
            visited.add(sorted_operands)
            
            for i in range(len(operands)) :
                for j in range(len(operands)) :
                    if i != j :
                        for op in self.operations :
                            new_operands = [operands[k] for k in range(len(operands)) if k != i and k != j]

                            if op == "+" :
                                if i > j :
                                    continue
                                new_operands.append(operands[i] + operands[j])
                            elif op == "-" :
                                if operands[i] >= operands[j] :
                                    new_operands.append(operands[i] - operands[j])
                                else :
                                    continue
                            elif op == "*" :
                                if i > j :
                                    continue
                                new_operands.append(operands[i] * operands[j])
                            elif op == "/" :
                                if operands[i] >= 0 and operands[j] > 0 and operands[i] % operands[j] == 0 :
                                    new_operands.append(operands[i] // operands[j])
                                else :
                                    continue
                            else :
                                raise NotImplementedError("Unsupported operation")
                            
                            if search(new_operands) :
                                return True
            
            return False
        
        return search(self.parameter["operands"])

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result["format"] :
            if processed_result["result"] is not None :
                if abs(processed_result["result"] - self.parameter["target"]) < self.epsilon :
                    return self.rewards["correct_answer"]
                else :
                    return self.rewards["wrong_answer"]
            else :
                return self.rewards["invalid_expression"]
        else :
            return self.rewards["wrong_format"]


class CountdownClose_Environment(Countdown_Environment) :
    def _prompt_template(self) -> str :
        return \
        r"""You are given the following numbers: {operands}
Your task is to create a valid mathematical expression whose result has the **minimal absolute difference** from the target number: **{target}**. Try your best to get as close to the target as possible.

Constraints:
- Use **each number exactly once**
- Only these basic arithmetic operations are allowed: {operations}
- You may use **parentheses** to define order of operations
- Your answer should be a **single valid expression**, with no extra explanation

Provide only the expression as your final answer."""


    def __init__(self,
                 wrong_format : float = -1.0, invalid_expression : float = -0.5, rewarding_strategy : str = "1/(1+|answer-target|)", rewarding_weight : float = 1.0,
                 **kwargs) :
        """
        Initialize the CountdownClose_Environment instance.

        Args:
            reward (dict): Dictionary of rewards for different evaluation results.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_expression" : invalid_expression,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
        }


    def _check_parameter(self) -> bool :
        if self.rewards["rewarding_strategy"] == "1/(1+|answer-target|)" and self.parameter["num_operands"] <= 6 :
            self.parameter["reference_result"] = None

            visited = set()
            def search(operands : List[float]) -> None :
                if len(operands) == 1 :
                    if self.parameter["reference_result"] is None or abs(operands[0] - self.parameter["target"]) < abs(self.parameter["reference_result"] - self.parameter["target"]) :
                        self.parameter["reference_result"] = operands[0]
                    return
                
                sorted_operands = tuple(sorted(map(lambda x : str(round(x, 5)), operands)))
                if sorted_operands in visited :
                    return
                visited.add(sorted_operands)
                
                for i in range(len(operands)) :
                    for j in range(len(operands)) :
                        if i != j :
                            for op in self.operations :
                                new_operands = [operands[k] for k in range(len(operands)) if k != i and k != j]

                                if op == "+" :
                                    if i > j :
                                        continue
                                    new_operands.append(operands[i] + operands[j])
                                elif op == "-" :
                                    new_operands.append(operands[i] - operands[j])
                                elif op == "*" :
                                    if i > j :
                                        continue
                                    new_operands.append(operands[i] * operands[j])
                                elif op == "/" :
                                    if operands[j] != 0 :
                                        new_operands.append(operands[i] / operands[j])
                                    else :
                                        continue
                                else :
                                    raise NotImplementedError("Unsupported operation")
                                
                                search(new_operands)
        
            search([float(operand) for operand in self.parameter["operands"]])
            assert self.parameter["reference_result"] is not None

            if self.rewards["rewarding_strategy"] == "1/(1+|answer-target|)" :
                self.passing_reward_threshold = self.rewards["rewarding_weight"] / (1 + abs(self.parameter["reference_result"] - self.parameter["target"]))
            else :
                assert False

        return True
        
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result["format"] :
            if processed_result["result"] is not None :
                if self.rewards["rewarding_strategy"] == "1/(1+|answer-target|)" :
                    return self.rewards["rewarding_weight"] / (1 + abs(processed_result["result"] - self.parameter["target"]))
                else :
                    raise NotImplementedError("Unsupported rewarding strategy")
            else :
                return self.rewards["invalid_expression"]
        else :
            return self.rewards["wrong_format"]