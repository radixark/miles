import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class BinaryAlternation_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a binary string of length {N}, consisting of `0`s and `1`s. It is 0-indexed: {string}

In one operation, you may **swap** the characters at indices `i` and `j` (0 ≤ i, j < {N}). Please transform the string into an **alternating binary string** (no two adjacent characters are the same) using the **minimum number of operations**.

**Output Format:** Each operation should be written on a single line in the format: `i j`, where `i` and `j` are the indices being swapped. Do **NOT** include backticks or quotes. Output one operation per line in the order they should be performed."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the BinaryAlternation_Environment instance.
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
        assert "zero_count" in self.parameter, "zero_count is required in parameter"
        zero_count = self.parameter["zero_count"]
        assert zero_count >= 2, "zero_count should be greater than or equal to 2"

        one_count = random.randint(zero_count - 1, zero_count + 1)

        string = ["0"] * zero_count + ["1"] * one_count
        random.shuffle(string)
        string = self.parameter["string"] = "".join(string)

        self.parameter["reference_answer"] = None


        def compute(should : str) -> List[str] :
            zero_to_one, one_to_zero = [], []
            for i, now in enumerate(string) :
                if now != should :
                    if now == "0" :
                        zero_to_one.append(i)
                    else :
                        one_to_zero.append(i)
                should = "1" if should == "0" else "0"
            assert len(zero_to_one) == len(one_to_zero), "zero_to_one and one_to_zero should have the same length"
            solution = []
            for i, j in zip(zero_to_one, one_to_zero) :
                solution.append("{} {}".format(i, j))
            return solution

        if zero_count >= one_count :
            self.parameter["reference_answer"] = compute("0")
        if one_count >= zero_count :
            candidate = compute("1")
            if self.parameter["reference_answer"] is None or len(candidate) < len(self.parameter["reference_answer"]) :
                self.parameter["reference_answer"] = candidate
        self.parameter["gold_answer"] = len(self.parameter["reference_answer"])
        self.parameter["reference_answer"] = "\n".join(self.parameter["reference_answer"])
    

    def _prompt_generate(self) -> str :
        string = self.parameter["string"]
        return self.prompt_template.format(N = len(string), string = string)


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            actions = []
            for line in answer.splitlines() :
                line = line.strip()
                if line :
                    actions.append(line.split())
                    action = actions[-1]
                    if len(action) != 2 :
                        return None
                    try :
                        action[0] = int(action[0])
                        action[1] = int(action[1])
                    except ValueError :
                        return None
            return actions
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            string = list(self.parameter["string"])
            for i, j in processed_result :
                if not (0 <= i < len(string) and 0 <= j < len(string)) :
                    return self.rewards["invalid_solution"]
                string[i], string[j] = string[j], string[i]
            string = "".join(string)
            if any(string[i] == string[i + 1] for i in range(len(string) - 1)) :
                return self.rewards["invalid_solution"]
            
            gold, answer = self.parameter["gold_answer"], len(processed_result)
            assert gold <= answer, "gold should be less than or equal to answer"

            if answer == 0 :
                return self.rewards["rewarding_weight"]
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]