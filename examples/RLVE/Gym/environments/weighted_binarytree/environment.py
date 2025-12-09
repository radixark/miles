import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class WeightedBinaryTree_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1040
    prompt_template = \
r"""You are given a binary tree with {N} nodes, labeled from 0 to {N_minus_1}.
The **in-order traversal** of the tree is: `0, 1, ..., {N_minus_1}` — that is, the in-order sequence is fixed in increasing order of node labels.

Each node `i` has an associated score `d_i` (where `0 ≤ i < {N}`), given as:
{scores}

The **score of a binary tree** is defined recursively as follows:
- `score(tree) = score(left_subtree) × score(right_subtree) + d_i`, where `i` is the root of the current subtree.
- If a subtree is **empty**, its score is defined to be `1`.
- If a node is a **leaf**, its score is simply `d_i` (ignore its empty subtrees).

Your task is to construct the binary tree that satisfies the above rules and has the **maximum possible score**, and then give its **pre-order traversal**.

Output Format:
Your final answer should be a single line containing the node labels in **pre-order traversal**, separated by **spaces**.
Example: `{all_node_sequence}` (do **NOT** include the backticks or quotes).
"""

    def __init__(self,
                 wrong_format : float = -1.0, not_permutation : float = -0.5, invalid_solution : float = 0.0, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "not_permutation" : not_permutation,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        "MAX_SCORE" in self.parameter, "MAX_SCORE is required in parameter"
        MAX_SCORE = self.parameter["MAX_SCORE"]
        assert MAX_SCORE >= 1, "MAX_SCORE should be greater than or equal to 1"

        scores = self.parameter["scores"] = [random.randint(1, MAX_SCORE) for _ in range(N)]

        dpF = [[0] * N for _ in range(N)]
        roots = [[None] * N for _ in range(N)]
        for i, score in enumerate(scores) :
            dpF[i][i] = score
            roots[i][i] = i
        for length in range(2, N + 1) :
            for i in range(N - length + 1) :
                j = i + length - 1
                for root in range(i, j + 1) :
                    left = dpF[i][root - 1] if i <= root - 1 else 1
                    right = dpF[root + 1][j] if root + 1 <= j else 1
                    if dpF[i][j] <= left * right + scores[root] :
                        dpF[i][j] = left * right + scores[root]
                        roots[i][j] = root
        self.parameter["gold"] = dpF[0][N - 1]

        def preorder(i, j) :
            if i > j :
                return []
            root = roots[i][j]
            return [root] + preorder(i, root - 1) + preorder(root + 1, j)
        self.parameter["reference_answer"] = " ".join(map(str, preorder(0, N - 1)))
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        scores = self.parameter["scores"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            scores="\n".join("d_{}={}".format(i, score) for i, score in enumerate(scores)),
            all_node_sequence = " ".join(map(str, range(N))),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            if len(processed_result) != self.parameter["N"] :
                return self.rewards["not_permutation"]
            if len(set(processed_result)) != self.parameter["N"] :
                return self.rewards["not_permutation"]
            for i in processed_result :
                if not (0 <= i < self.parameter["N"]) :
                    return self.rewards["not_permutation"]
            
            def get_score(inorder_l : int, inorder_r : int, preorder : list[int]) -> Optional[int] :
                # The in-order traversal sequence is [inorder_l, inorder_r]
                # The pre-order traversal sequence is preorder
                assert len(preorder) == inorder_r - inorder_l + 1, "preorder should have the same length as inorder"

                root = preorder[0]
                if inorder_l <= root <= inorder_r :
                    if inorder_l == inorder_r :
                        return self.parameter["scores"][root]
                    left = get_score(inorder_l, root - 1, preorder[1 : 1 + (root - 1 - inorder_l) + 1]) if inorder_l <= root - 1 else 1
                    right = get_score(root + 1, inorder_r, preorder[1 + (root - 1 - inorder_l) + 1 :]) if root + 1 <= inorder_r else 1
                    if left is not None and right is not None :
                        return left * right + self.parameter["scores"][root]
                    else :
                        return None
                else :
                    return None
            answer = get_score(0, self.parameter["N"] - 1, processed_result)
            if answer is None :
                return self.rewards["invalid_solution"]

            assert answer <= self.parameter["gold"], "answer should be less than or equal to gold"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / self.parameter["gold"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == self.parameter["gold"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]