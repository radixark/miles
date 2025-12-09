import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinInorderBinaryTree_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given {N} nodes numbered from 1 to {N}, along with the following edges (for each edge, the parent–child direction is not specified):
{edges}

Please construct a valid **binary tree** using all these edges. Among all possible binary trees that can be formed, choose the one whose **inorder traversal** is lexicographically smallest. Output a single line containing {N} space-separated integers — the inorder traversal of the chosen binary tree."""
    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the MinInorderBinaryTree_Environment instance.
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
        assert N >= 4, "N should be greater than or equal to 4"

        edges = self.parameter["edges"] = []
        def construct(nodes : List[int]) -> int :
            random.shuffle(nodes)
            root = nodes[0]
            left_size = random.randint(0, len(nodes) - 1)
            right_size = len(nodes) - 1 - left_size
            if left_size > 0 :
                left_root = construct(nodes[1 : 1 + left_size])
                edges.append((min(root, left_root), max(root, left_root)))
            if right_size > 0 :
                right_root = construct(nodes[1 + left_size : ])
                edges.append((min(root, right_root), max(root, right_root)))
            return root
        construct(list(range(1, N + 1)))
        random.shuffle(edges)

        assert len(edges) == len(set(edges)) == N - 1, "edges should be unique and of size N-1"
        assert all(1 <= u < v <= N for u, v in edges), "edges should be between 1 and N"


        G = [[] for _ in range(N + 1)]
        SON = [[] for _ in range(N + 1)]
        FA = [0] * (N + 1)
        HEAD = [0] * (N + 1)

        for u, v in edges :
            G[u].append(v)
            G[v].append(u)

        # Choose a start node FIR: the smallest index (scanning from N down to 1) whose degree != 3
        FIR = 0
        for i in range(N, 0, -1):
            if (len(G[i]) ^ 3) != 0:
                FIR = i

        def build(start):
            """Equivalent to dfs(u) in C++: builds SON and HEAD given a root 'start' using FA as parent array."""
            # Clear SON
            for idx in range(1, N + 1):
                SON[idx] = []
            order = []
            FA[start] = 0
            stack = [start]
            while stack:
                u = stack.pop()
                order.append(u)
                for v in G[u]:
                    if v != FA[u]:
                        SON[u].append(v)
                        FA[v] = u
                        stack.append(v)
            # Post-order compute HEAD
            for u in reversed(order):
                if len(SON[u]) == 0:
                    HEAD[u] = u
                elif len(SON[u]) == 1:
                    c = SON[u][0]
                    HEAD[u] = u if u < HEAD[c] else HEAD[c]
                else:
                    a, b = SON[u][0], SON[u][1]
                    HEAD[u] = HEAD[a] if HEAD[a] < HEAD[b] else HEAD[b]

        # First build from FIR
        build(FIR)

        # dfs1(u): determine the root rt
        u = FIR
        while True:
            if len(SON[u]) == 0:
                rt = u
                break
            elif len(SON[u]) == 1:
                c = SON[u][0]
                if HEAD[c] < c:
                    rt = u
                    break
                else:
                    u = c
            else:  # len == 2
                a, b = SON[u][0], SON[u][1]
                if HEAD[a] < HEAD[b]:
                    u = b
                else:
                    u = a

        # Rebuild with chosen root
        FA[rt] = 0
        build(rt)

        # dfs2(u): inorder traversal with tie-breaking rules to get lexicographically smallest sequence
        ans = []
        stack = [(rt, 'go')]
        while stack:
            node, typ = stack.pop()
            if typ == 'emit':
                ans.append(node)
                continue
            # typ == 'go'
            if len(SON[node]) == 0:
                ans.append(node)
            elif len(SON[node]) == 1:
                c = SON[node][0]
                if node < HEAD[c]:
                    # output node, then child
                    stack.append((c, 'go'))
                    stack.append((node, 'emit'))
                else:
                    # child, then node
                    stack.append((node, 'emit'))
                    stack.append((c, 'go'))
            else:
                a, b = SON[node][0], SON[node][1]
                # choose left/right based on HEAD comparison
                if HEAD[a] < HEAD[b]:
                    left, right = a, b
                else:
                    left, right = b, a
                # inorder: left, node, right => push in reverse
                stack.append((right, 'go'))
                stack.append((node, 'emit'))
                stack.append((left, 'go'))

        self.parameter["gold_answer"] = ans
        self.parameter["reference_answer"] = " ".join(map(str, ans))
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[List[int]] :
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

            if len(processed_result) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if set(processed_result) != set(range(1, self.parameter["N"] + 1)) :
                return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["gold_answer"], processed_result)) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]