import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class FBI_BinaryTree_Environment(VerifiableEnvironment) : # Source: https://www.luogu.com.cn/problem/P1087
    prompt_template = \
r"""We classify binary strings made up of only `0` and `1` into three types:
- A string consisting of only `0`s is called a **B-string**.
- A string consisting of only `1`s is called an **I-string**.
- A string that contains both `0` and `1` is called an **F-string**.

An **FBI tree** is a binary tree where each node is labeled as either F, B, or I, based on the type of the substring it represents.
Given a binary string `S`, construct an FBI tree `T` using the following recursive rules:
1. The **root node** corresponds to the entire string `S`, and its type is determined using the rules above.
2. If the length of `S` is greater than 1, divide `S` exactly in half into two equal substrings: `S₁` (left) and `S₂` (right). Recursively build the **left subtree** from `S₁`, and the **right subtree** from `S₂`.

Your task is to construct the FBI tree from the following binary string of length 2^{N}:
{string}

Then, output the **postorder traversal** of the tree — a string consisting of the node types in postorder (left, right, root).

Output Format:
Your output should be a single line containing the postorder traversal of the tree. Each node type (F, B, or I) should appear **without any separators**.
Example: `{all_B_answer}` (do **NOT** include the backticks or quotes).
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 probability_same_as_before : float = 0.7,
                 **kwargs) :
        """
        Initialize the FBI_BinaryTree_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
        
        self.probability_same_as_before = probability_same_as_before
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 1, "N should be greater than or equal to 1"

        string = [random.randint(0, 1)]
        for i in range(1, 2 ** N) :
            if random.random() < self.probability_same_as_before :
                string.append(string[i - 1])
            else :
                string.append(random.randint(0, 1))
        string = self.parameter["string"] = "".join(map(str, string))
        assert len(self.parameter["string"]) == (2**N), "string length should be {}".format(2**N)

        def get_postorder(l, r) :
            if l == r :
                if string[l] == "0" :
                    return "B"
                else :
                    return "I"
            left, right = get_postorder(l, (l + r) // 2), get_postorder((l + r) // 2 + 1, r)
            if left[-1] == "B" and right[-1] == "B" :
                root = "B"
            elif left[-1] == "I" and right[-1] == "I" :
                root = "I"
            else :
                root = "F"
            return left + right + root
        self.parameter["reference_answer"] = get_postorder(0, 2**N - 1)
        assert len(self.parameter["reference_answer"]) == (2**(N + 1) - 1), "reference_answer length should be {}".format(2**(N + 1) - 1)
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            string = self.parameter["string"],
            all_B_answer = "B" * len(self.parameter["reference_answer"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            answer = answer.strip()
            return answer
        else :
            return None
    
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if len(processed_result) != len(self.parameter["reference_answer"]) :
                return self.rewards["invalid_solution"]
            for char in processed_result :
                if char not in ("F", "B", "I") :
                    return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(float(a == b) for a, b in zip(self.parameter["reference_answer"], processed_result)) / len(self.parameter["reference_answer"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * all(a == b for a, b in zip(self.parameter["reference"], processed_result))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]