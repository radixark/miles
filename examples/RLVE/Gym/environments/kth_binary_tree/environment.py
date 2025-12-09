import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Kth_BinaryTree_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2274
    prompt_template = \
r"""A binary tree is assigned a unique non-negative integer index based on the following rules:

1. The empty tree has index 0; a single-node tree has index 1.
2. Among all binary trees, those with fewer nodes have smaller indices.
3. For two distinct binary trees A and B with the same number of nodes:
   - If the left subtree of A has a smaller index than that of B, then A has a smaller index.
   - If their left subtree indices are equal, then the tree with the smaller right subtree index has the smaller overall index.
4. Indices are continuous and unique: each non-negative integer maps to exactly one binary tree, and vice versa.

Find the binary tree with index {N} and output its postorder traversal using the following format:
- A single-node tree is represented as `X`.
- For a tree with left subtree L and right subtree R (represented as L' and R' respectively), the postorder is `(L')X(R')`.
- If the left subtree is empty, omit its parentheses: `X(R')`.
- If the right subtree is empty, omit its parentheses: `(L')X`.

**Output Format:** Your output should be a single line containing the postorder traversal.
Example: `((X)X(X))X` (do **NOT** include quotes or backticks; this is the binary tree with index 20)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the Kth_BinaryTree_Environment instance.
        """

        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 1, "MAX_N should be greater than or equal to 1"

        N = self.parameter["N"] = random.randint(1, MAX_N)


        ordinal = N + 1

        f = [1, 1]
        g = [1, 2]

        i = 2
        while g[-1] < ordinal :
            fi = 0
            for j in range(i) :
                fi += f[j] * f[i - j - 1]
            f.append(fi)
            g.append(g[-1] + fi)
            i += 1

        def build(order, wrap) :
            if order <= 1:
                return ""
            s = []
            if wrap :
                s.append("(")

            size = next(idx for idx, gi in enumerate(g) if order <= gi)
            rest = order - (g[size - 1] if size > 0 else 0)

            for left_nodes in range(size) :
                right_nodes = size - 1 - left_nodes
                block = f[left_nodes] * f[right_nodes]
                if rest <= block :
                    left_rank = (rest - 1) // f[right_nodes] + 1
                    right_rank = rest - (left_rank - 1) * f[right_nodes]

                    left_ord = left_rank + (g[left_nodes - 1] if left_nodes > 0 else 0)
                    right_ord = right_rank + (g[right_nodes - 1] if right_nodes > 0 else 0)

                    s.append(build(left_ord, True))
                    s.append("X")
                    s.append(build(right_ord, True))
                    break
                rest -= block

            if wrap :
                s.append(")")
            return "".join(s)

        self.parameter["reference_answer"] = build(ordinal, False)
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"])
    

    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            answer = answer.strip()
            return answer
        else :
            return None
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if not all(c in "X()" for c in processed_result) :
                return self.rewards["invalid_solution"]
            
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]