import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class StringReversalConstruction_Environment(VerifiableEnvironment):
    prompt_template = \
r"""A code lock is installed on a safe. The lock has a screen that displays a string of {n} lowercase Latin letters. Initially, the screen displays string "{s}". The safe will open when string "{t}" is displayed on the screen.

The string on the screen can be changed using the operation "shift x". To apply this operation, you choose an integer x from 0 to {n} (including 0 and {n}). After that, the current string p = α + β changes to β^R + α, where the length of β is x, and the length of α is {n} - x. In other words, the suffix of length x of string p is reversed and moved to the beginning of the string (+ means string concatenation and β^R means the reverse of β). For example, after the operation "shift 4" the string "abcacb" will be changed to "bcacab", since α = "ab", β = "cacb", β^R = "bcac".

Find a way to open the safe, using no more than {max_k} operations.

Your response should only contain the solution in the following format: a single line containing k numbers x_i corresponding to the operations "shift x_i" (0 ≤ x_i ≤ {n}) in the order in which they should be applied (separated by spaces), where k is the number of operations."""

    def __init__(self,
                 wrong_format: float = -1.0, 
                 invalid_solution: float = -0.5, 
                 incorrect_solution: float = 0.0,
                 correct_solution: float = 1.0,
                 **kwargs):
        """
        Initialize the StringReversalConstruction_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "incorrect_solution": incorrect_solution,
            "correct_solution": correct_solution,
        }

    def _apply_shift_operation(self, s: str, x: int) -> str:
        """Apply shift x operation to string s"""
        n = len(s)
        assert 0 <= x <= n, "x must be in the range [0, n]"
        if x == 0:
            return s
        if x == n:
            return s[::-1]
        
        alpha = s[:-x]  # first n-x characters
        beta = s[-x:]   # last x characters
        beta_reversed = beta[::-1]
        
        return beta_reversed + alpha

    def _generate(self) -> None:
        assert "n" in self.parameter, "n is required in parameter"
        
        n = self.parameter["n"]
        
        # Generate initial string s
        s = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(n))
        
        # Generate target string t by applying random operations to s
        # This ensures there's always a valid solution
        t = s
        num_operations = random.randint(1, max(1, n // 2))
        operations = []
        
        for _ in range(num_operations):
            x = random.randint(1, n)
            t = self._apply_shift_operation(t, x)
            operations.append(x)
        
        self.parameter["n"] = n
        self.parameter["s"] = s
        self.parameter["t"] = t
        self.parameter["reference_answer"] = " ".join(map(str, operations))
        self.parameter["max_k"] = 3 * n

    def _prompt_generate(self) -> str:
        return self.prompt_template.format(
            n=self.parameter["n"],
            s=self.parameter["s"],
            t=self.parameter["t"],
            max_k=self.parameter["max_k"]
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

    def scorer(self, output: str) -> float:
        processed_result = self.processor(output)
        
        if processed_result is None:
            return self.rewards["wrong_format"]
        
        operations = processed_result
        
        # Check if number of operations exceeds limit
        if len(operations) > self.parameter["max_k"]:
            return self.rewards["invalid_solution"]
        if not all(0 <= op <= self.parameter["n"] for op in operations):
            return self.rewards["invalid_solution"]
        
        # Simulate the operations
        current_s = self.parameter["s"]
        target_t = self.parameter["t"]
        for op in operations:
            current_s = self._apply_shift_operation(current_s, op)
        if current_s == target_t:
            return self.rewards["correct_solution"]
        else:
            return self.rewards["incorrect_solution"]