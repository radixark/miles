import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class VisibleLine_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3194
    prompt_template = \
r"""You are given {N} lines on the 2D plane:
{lines}

We say a line is **visible** if any portion of it can be seen when viewed from y = +∞ (i.e., looking vertically downward). That is, a line is visible if there exists at least one x-coordinate such that this line lies on top (i.e., has the maximum y-value) at that x among all lines.

**Output Format:** A single line containing the indices of all visible lines, in any order, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(intersection/union)^beta", rewarding_beta : float = 5.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the VisibleLine_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        lines = set()
        while len(lines) < N :
            Ai, Bi = random.randint(-N, +N), random.randint(-N, +N)
            if (Ai, Bi) not in lines :
                lines.add((Ai, Bi))
        self.parameter["lines"] = lines = list(lines)
        random.shuffle(lines)


        P = []
        for i, (A, B) in enumerate(lines):
            P.append((A, B, i))  # store 1-based id for output

        # Sort by slope A ascending, and for ties by intercept B descending
        P.sort(key=lambda x: (x[0], -x[1]))

        # Build the "upper hull" of visible lines
        BIN = []
        prevA = None
        for A, B, idx in P:
            # skip duplicate slopes (only keep the one with highest intercept)
            if A == prevA:
                continue
            prevA = A

            # While the last segment and the new point make a non-left turn,
            # pop the last line (it's covered)
            while len(BIN) >= 2:
                A1, B1, _ = BIN[-2]
                A2, B2, _ = BIN[-1]
                # cross product of vectors (A2-A1, B2-B1) and (A-A2, B-B2)
                if (A2 - A1) * (B - B2) - (B2 - B1) * (A - A2) >= 0:
                    BIN.pop()
                else:
                    break

            BIN.append((A, B, idx))

        # Sort visible lines by original input order (their ids)
        BIN.sort(key=lambda x: x[2])

        # Output the ids with a trailing space after each, including the last
        self.parameter["gold_answer"] = [idx for A, B, idx in BIN]
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["gold_answer"]))
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            lines = "\n".join("Line {}: y = {}x + {}".format(i, A, B) for i, (A, B) in enumerate(self.parameter["lines"])),
        )


    def _process(self, answer : Optional[str]) -> Optional[set] :
        if answer is not None :
            answer = answer.strip()
            try :
                return set(map(int, answer.split()))
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, set), "processed_result should be a list"

            answer = processed_result
            if not all(0 <= x < self.parameter["N"] for x in answer) :
                return self.rewards["wrong_format"]
            gold = set(self.parameter["gold_answer"])

            if self.rewards["rewarding_strategy"] == "(intersection/union)^beta" :
                intersection = len(answer & gold)
                union = len(answer | gold)
                return ((intersection / union) ** self.rewards["rewarding_beta"]) * self.rewards["rewarding_weight"]
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]