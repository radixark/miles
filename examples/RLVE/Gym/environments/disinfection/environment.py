import random
from typing import Optional, List, Tuple
from Gym.environment import VerifiableEnvironment


class Disinfection_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a 3D cube of dimensions {A} × {B} × {C} (0-indexed). Some cells in the cube contain the value 1, and the rest are 0. The coordinates of the cells with value 1 are:
{one_coordinates}

In one operation, you may select a contiguous sub-cube defined by ranges: x ∈ [x1, x2) y ∈ [y1, y2) z ∈ [z1, z2), where 0 ≤ x1 < x2 ≤ {A}, 0 ≤ y1 < y2 ≤ {B}, and 0 ≤ z1 < z2 ≤ {C}. This operation sets **all** values in the sub-cube to 0. The cost of this operation is defined as min(x2 - x1, y2 - y1, z2 - z1).
Please set **all** values in the cube to 0 using a set of such operations with the **minimum total cost**.

**Output Format:** Output multiple lines. Each line should contain six integers `x1 x2 y1 y2 z1 z2` (do **NOT** include quotes or backticks), separated by spaces, representing one operation."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Disinfection_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "unsuccessful_solution" : unsuccessful_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_A_B_C" in self.parameter, "MAX_A_B_C is required in parameter"
        MAX_A_B_C = self.parameter["MAX_A_B_C"]
        assert MAX_A_B_C >= 2, "MAX_A_B_C should be greater than or equal to 2"

        while True :
            A, B, C = self.parameter["A"], self.parameter["B"], self.parameter["C"] = random.randint(1, MAX_A_B_C), random.randint(1, MAX_A_B_C), random.randint(1, MAX_A_B_C)
            if A != 1 or B != 1 or C != 1 :
                break
        subA, subB, subC = random.sample(range(A), random.randint(1, A)), random.sample(range(B), random.randint(1, B)), random.sample(range(C), random.randint(1, C))
        one_coordinates = self.parameter["one_coordinates"] = random.sample([(x, y, z) for x in subA for y in subB for z in subC], random.randint(1, len(subA) * len(subB) * len(subC)))
        random.shuffle(one_coordinates)


        def solve_one_case() -> None:
            DIMS = [A, B, C]

            # ---------- find the shortest axis ----------
            pos = DIMS.index(min(DIMS))        # 0, 1 or 2
            SMALL = DIMS[pos]                  # length of the short axis

            # Decide which of the remaining two axes is "left" (U side of the
            # bipartite graph) and which is "right" (V side).  The original
            # code always put the first coordinate **not equal to `pos`** on
            # the left, so we do the same.
            if pos == 0:
                left_len, right_len = B, C     # U = j, V = k
            elif pos == 1:
                left_len, right_len = A, C     # U = i, V = k
            else:                              # pos == 2
                left_len, right_len = A, B     # U = i, V = j

            CNT = max(left_len, right_len)     # array size used in the C++ code

            # ---------- build the 3-D grid and the edge list ----------
            GRID = [[[0] * C for _ in range(B)] for _ in range(A)]
            adjacency = [[] for _ in range(CNT)]    # list[ list[ (v, layer) ] ]

            # helper to add an (undirected) edge with its layer index
            def add_edge(u: int, v: int, layer: int) -> None:
                adjacency[u].append((v, layer))

            for i, j, k in one_coordinates:
                if pos == 0:          # short axis = i
                    u, v, layer = j, k, i
                elif pos == 1:        # short axis = j
                    u, v, layer = i, k, j
                else:                 # short axis = k
                    u, v, layer = i, j, k
                add_edge(u, v, layer)

            # ---------- variables used in the recursive search ----------
            SEL = [False] * SMALL             # which layers of the short axis are chosen
            VIS = [0] * CNT                   # time-stamped visitation array
            MATCH = [-1] * CNT                # right-side match array  (my[ ] in C++)
            cur_time = 0                      # global DFS clock
            best_answer = [10 ** 9]           # wrapped in list for closure mutability

            # ---------- depth-first search for augmenting paths ----------
            def dfs(u: int) -> bool:
                nonlocal cur_time
                for v, lay in adjacency[u]:
                    if SEL[lay]:                   # layer already paid for
                        continue
                    if VIS[v] == cur_time:         # already visited in this search
                        continue
                    VIS[v] = cur_time
                    if MATCH[v] == -1 or dfs(MATCH[v]):
                        MATCH[v] = u
                        return True
                return False

            # ---------- run a Hungarian style matching on surviving edges ----------
            def run_matching(paid: int) -> int:
                """Return paid + |maximum matching|   (early-terminate if ≥ best)."""
                nonlocal cur_time, MATCH
                MATCH = [-1] * CNT
                matched = 0
                for u in range(CNT):
                    cur_time += 1
                    if dfs(u):
                        matched += 1
                        if paid + matched >= best_answer[0]:
                            return paid + matched   # prune
                return paid + matched

            # ---------- enumerate every subset of the short axis ----------
            def enumerate_layers(depth: int, paid: int) -> None:
                if depth == SMALL:                 # considered all layers
                    cost = run_matching(paid)
                    if cost < best_answer[0]:
                        best_answer[0] = cost
                    return
                # Case 1: pay for this layer
                SEL[depth] = True
                enumerate_layers(depth + 1, paid + 1)
                # Case 2: do not pay
                SEL[depth] = False
                enumerate_layers(depth + 1, paid)

            enumerate_layers(0, 0)
            self.parameter["gold_answer"] = best_answer[0]
            assert self.parameter["gold_answer"] > 0, "Gold answer should be greater than 0"
        solve_one_case()
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            A = self.parameter["A"],
            B = self.parameter["B"],
            C = self.parameter["C"],
            one_coordinates = "\n".join("({},{},{})".format(x, y, z) for x, y, z in self.parameter["one_coordinates"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[List[Tuple[int, int, int, int, int, int]]] :
        if answer is not None :
            answer = answer.strip()
            try :
                matrix = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        matrix.append(tuple(map(int, line.split())))
                if not all(len(row) == 6 for row in matrix) :
                    return None
                return matrix
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            answer, gold = 0, self.parameter["gold_answer"]
            disinfected = [[[False] * self.parameter["C"] for _ in range(self.parameter["B"])] for _ in range(self.parameter["A"])]
            for x1, x2, y1, y2, z1, z2 in processed_result :
                if not (0 <= x1 < x2 <= self.parameter["A"]) :
                    return self.rewards["invalid_solution"]
                if not (0 <= y1 < y2 <= self.parameter["B"]) :
                    return self.rewards["invalid_solution"]
                if not (0 <= z1 < z2 <= self.parameter["C"]) :
                    return self.rewards["invalid_solution"]
                for x in range(x1, x2) :
                    for y in range(y1, y2) :
                        for z in range(z1, z2) :
                            disinfected[x][y][z] = True
                answer += min(x2 - x1, y2 - y1, z2 - z1)
            for x, y, z in self.parameter["one_coordinates"] :
                if not disinfected[x][y][z] :
                    return self.rewards["unsuccessful_solution"]
            assert gold <= answer, "Gold answer should be less than or equal to the answer"

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]