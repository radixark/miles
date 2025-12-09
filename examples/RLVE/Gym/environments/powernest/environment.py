import random
from typing import Dict, Optional, Any
from Gym.environment import VerifiableEnvironment


class PowerNest_Environment(VerifiableEnvironment) : # Source: https://www.luogu.com.cn/problem/P1010
    prompt_template = \
r"""You are given a **positive integer** `{number}`.

Every positive integer can be represented as a **sum of powers of 2**. For example:
137 = 2^7 + 2^3 + 2^0

We adopt the following format:
- A power expression like a^b should be written as `a(b)`
- So, 137 can be written as: `2(7)+2(3)+2(0)`

Now, each exponent (like `7`, `3`, `0`) can itself be expressed as a sum of powers of 2, recursively applying the same rule:
- 7 = 2^2 + 2 + 2^0 → 2(2)+2+2(0)
- 3 = 2 + 2^0 → 2+2(0)

So the final expression for 137 becomes:
`2(2(2)+2+2(0))+2(2+2(0))+2(0)`

Another example:
1315 = 2^10 + 2^8 + 2^5 + 2 + 1
Final form: `2(2(2+2(0))+2)+2(2(2+2(0)))+2(2(2)+2(0))+2+2(0)`

---

Your task is to write the given number `{number}` in this **power-of-two expression form**, following the rules above.

Output Format:
Your final answer should be just the final expression, e.g. `2(2(2+2(0))+2)+2(2(2+2(0)))+2(2(2)+2(0))+2+2(0)` (do **NOT** include the backticks or quotes).
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the PowerNest_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "max_number" in self.parameter, "max_number is required in parameter"
        max_number = self.parameter["max_number"]
        assert max_number >= 1, "max_number should be greater than or equal to 1"

        self.parameter["number"] = random.randint(1, max_number)
        
        n2expression = {}
        def convert_to_powernest(n) :
            assert n > 0, "n should be greater than 0"
            if n in n2expression :
                return n2expression[n]
            power = 0
            result = []
            while n :
                if n & 1 :
                    if power == 0 :
                        result.append("2(0)")
                    elif power == 1 :
                        result.append("2")
                    else :
                        result.append("2({})".format(convert_to_powernest(power)))
                n //= 2
                power += 1
            result.reverse()
            n2expression[n] = "+".join(result)
            return n2expression[n]
        self.parameter["reference_answer"] = convert_to_powernest(self.parameter["number"])
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(number = self.parameter["number"])
    

    def _process(self, answer : Optional[str]) -> Dict[str, Any] :
        if answer is not None :
            answer = answer.strip()
            if answer == self.parameter["reference_answer"] :
                return {"format" : True, "validation" : True, "answer" : answer}
            else :
                def check_powernest(expression) :
                    if expression == "" :
                        return False
                    
                    intervals = []
                    stack_count = 0
                    for i, char in enumerate(expression) :
                        if char == "(" :
                            stack_count += 1
                        elif char == ")" :
                            if stack_count > 0 :
                                stack_count -= 1
                            else :
                                return False
                        elif char == "+" :
                            if stack_count == 0 :
                                if not intervals :
                                    intervals.append((0, i))
                                else :
                                    intervals.append((intervals[-1][1] + 1, i))
                        else :
                            pass
                    if stack_count != 0 :
                        return False
                    
                    if intervals :
                        intervals.append((intervals[-1][1] + 1, len(expression)))
                        for interval in intervals :
                            if interval[0] < interval[1] :
                                if not check_powernest(expression[interval[0] : interval[1]]) :
                                    return False
                            else :
                                return False
                        return True
                    else :
                        if expression == "2" :
                            return True
                        elif expression.startswith("2(") and expression.endswith(")") :
                            if expression[2 : -1] == "0" :
                                return True
                            return check_powernest(expression[2 : -1])
                        else :
                            return False
                
                return {"format" : True, "validation" : check_powernest(answer), "answer" : answer}
        else :
            return {"format" : False}

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result["format"] :
            if processed_result["validation"] :
                if processed_result["answer"] == self.parameter["reference_answer"] :
                    return self.rewards["correct_answer"]
                else :
                    return self.rewards["wrong_answer"]
            else :
                return self.rewards["invalid_solution"]
        else :
            return self.rewards["wrong_format"]