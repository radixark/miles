import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class PreorderTraversal_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a binary tree with nodes labeled from 0 to {N_minus_1}.

Its **in-order traversal** sequence is: {inorder_traversal}
Its **post-order traversal** sequence is: {postorder_traversal}

Your task is to reconstruct the tree and output its **pre-order traversal** sequence.

Output Format:
Your final answer should be a single line containing the pre-order traversal, with node labels separated by **spaces**.
Example: `{all_node_sequence}` (do **NOT** include the backticks or quotes).
"""
    def __init__(self,
                 wrong_format : float = -1.0, wrong_length : float = 0.0, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the PreorderTraversal_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_length" : wrong_length,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        nodes = list(range(N))
        random.shuffle(nodes)
        def build(nodes) :
            if not nodes :
                return None
            root_index = random.randint(0, len(nodes) - 1)
            return {
                "root" : nodes[root_index],
                "left" : build(nodes[: root_index]),
                "right" : build(nodes[root_index + 1 :]),
            }
        tree = build(nodes)

        def preorder_traversal(node) :
            if node is None :
                return []
            return [node["root"]] + preorder_traversal(node["left"]) + preorder_traversal(node["right"])
        def inorder_traversal(node) :
            if node is None :
                return []
            return inorder_traversal(node["left"]) + [node["root"]] + inorder_traversal(node["right"])
        def postorder_traversal(node) :
            if node is None :
                return []
            return postorder_traversal(node["left"]) + postorder_traversal(node["right"]) + [node["root"]]
        self.parameter["inorder_traversal"] = inorder_traversal(tree)
        self.parameter["postorder_traversal"] = postorder_traversal(tree)
        
        self.parameter["preorder_traversal"] = preorder_traversal(tree)
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["preorder_traversal"]))
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N_minus_1 = N - 1,
            inorder_traversal = " ".join(map(str, self.parameter["inorder_traversal"])),
            postorder_traversal = " ".join(map(str, self.parameter["postorder_traversal"])),
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
                return self.rewards["wrong_length"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(float(a == b) for a, b in zip(self.parameter["preorder_traversal"], processed_result)) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * all(a == b for a, b in zip(self.parameter["preorder_traversal"], processed_result))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]