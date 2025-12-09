import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SingleStackSorting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a queue of integers containing `{N}` elements in increasing order from `0` (at the front) to `{N_minus_1}` (at the back). You also have an empty stack `S` and an initially empty output sequence. You may perform the following operations:
- `a`: Pop the front element of the queue and push it onto the stack `S`.
- `b`: Pop the top element from the stack `S` and append it to the output sequence.

Please produce the following target output sequence:  
{sequence}

Please output a valid sequence of operations (a string consisting of the characters `a` and `b` only) that transforms the initial queue into the given output sequence using the rules above.

**Output Format:** A single line containing the sequence of operations (`a` and `b` only), with no spaces or extra characters."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the SingleStackSorting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        operation_distribution = [random.randint(1, N) for _ in range(2)]
        operation_distribution = [weight / sum(operation_distribution) for weight in operation_distribution]

        self.parameter["reference_answer"] = ""
        
        S = []
        output_sequence = self.parameter["output_sequence"] = []
        queue_front = 0
        while len(output_sequence) < N :
            operation = random.choices(["a", "b"], weights = operation_distribution, k = 1)[0]
            if operation == "a" and queue_front < N :
                self.parameter["reference_answer"] += "a"
                S.append(queue_front)
                queue_front += 1
            elif operation == "b" and S :
                self.parameter["reference_answer"] += "b"
                output_sequence.append(S.pop())
        assert len(self.parameter["reference_answer"]) == N * 2, "reference_answer should have length 2 * N"
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            sequence = " ".join(map(str, self.parameter["output_sequence"])),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            return answer.strip()
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            S = []
            output_sequence = []
            queue_front = 0

            for operation in processed_result :
                if operation == "a" :
                    if queue_front >= self.parameter["N"] :
                        return self.rewards["invalid_solution"]
                    S.append(queue_front)
                    queue_front += 1
                elif operation == "b" :
                    if not S :
                        return self.rewards["invalid_solution"]
                    output_sequence.append(S.pop())
                else :
                    return self.rewards["wrong_format"]
            
            if len(output_sequence) != self.parameter["N"] :
                return self.rewards["invalid_solution"]

            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["output_sequence"], output_sequence)) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["output_sequence"] == output_sequence)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]