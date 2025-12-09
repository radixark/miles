import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class BlockImage_Environment(VerifiableEnvironment) : # Source: https://www.luogu.com.cn/problem/P1058
    prompt_template = \
r"""You are given a {M} × {N} rectangular grid, where each cell represents a stack of identical cube blocks. Each cube has size 1 × 1 × 1, and no rotation or flipping is allowed — all cubes are placed in the same orientation.
You are given a matrix representing the number of cubes stacked on each cell in the grid (the integer at row i and column j indicates how many cube blocks are stacked on the cell located at row i, column j):
{matrix}

The visual representation of a **single cube** follows this fixed format:

$$
\def\arraystretch{1e-10}
\begin{aligned}
&\verb!  +---+!\\
&\verb! /   /|!\\
&\verb!+---+ |!\quad\textsf{height}\\
&\verb!|   | +!\\
&\verb!|   |/ !\quad\textsf{width}\\
&\verb!+---+  !\\
& \quad\textsf{length}
\end{aligned}
$$

Each `+` represents a corner, `-` spans the cube’s length, `/` shows depth (width), and `|` shows height. Empty space in the final drawing should be represented using `.`.

The 3D isometric projection follows specific stacking rules:

- **Two cubes side by side (left/right):**
$$
\def\arraystretch{1e-10}
\begin{aligned}
\verb!..+---+---+!\\
\verb!./   /   /|!\\
\verb!+---+---+ |!\\
\verb!|   |   | +!\\
\verb!|   |   |/.!\\
\verb!+---+---+..!\\
\end{aligned}
$$

- **Two cubes stacked vertically (top/bottom):**
$$
\def\arraystretch{1e-10}
\begin{aligned}
\verb!..+---+!\\
\verb!./   /|!\\
\verb!+---+ |!\\
\verb!|   | +!\\
\verb!|   |/|!\\
\verb!+---+ |!\\
\verb!|   | +!\\
\verb!|   |/.!\\
\verb!+---+..!\\
\end{aligned}
$$

- **Two cubes front/back (depth):**
$$
\def\arraystretch{1e-10}
\begin{aligned}
\verb!....+---+!\\
\verb!.../   /|!\\
\verb!..+---+ |!\\
\verb!./   /| +!\\
\verb!+---+ |/.!\\
\verb!|   | +..!\\
\verb!|   |/...!\\
\verb!+---+....!\\
\end{aligned}
$$

The bottom-left corner of the lowest cube in cell ({M}, 1) (bottom row, first column) should align with the bottom-left of the entire drawing.

**Output Format:**
Your final output should be a string matrix of dimensions K × L (i.e., it has K lines separated by line breaks, with each line containing exactly L characters), where K is the number of rows and L is the number of columns **required to draw the 3D structure correctly** according to the rules above.

---

**Example 1**

When the rectangular grid is 1 × 2, and the number of cubes in each cell is as follows:
1 3

The output is (do **NOT** include the backticks or quotes — use the format below exactly):
```
......+---+
...../   /|
....+---+ |
....|   | +
....|   |/|
....+---+ |
..+-|   | +
./  |   |/|
+---+---+ |
|   |   | +
|   |   |/.
+---+---+..
```

---

**Example 2**

When the rectangular grid is 3 × 4, and the number of cubes in each cell is as follows:
2 2 1 2
2 2 1 1
3 2 1 2

The output is (do **NOT** include the backticks or quotes — use the format below exactly):
```
......+---+---+...+---+
..+---+  /   /|../   /|
./   /|-+---+ |.+---+ |
+---+ |/   /| +-|   | +
|   | +---+ |/+---+ |/|
|   |/   /| +/   /|-+ |
+---+---+ |/+---+ |/| +
|   |   | +-|   | + |/.
|   |   |/  |   |/| +..
+---+---+---+---+ |/...
|   |   |   |   | +....
|   |   |   |   |/.....
+---+---+---+---+......
```
"""

    def __init__(self,
                 max_height : int = 5,
                 wrong_format : float = -1.0, invalid_answer : int = -0.5, wrong_size : int = 0.0, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 2.0,
                 **kwargs) :
        """
        Initialize the BlockImage_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_height = max_height

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "wrong_size" : wrong_size,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "MAX_M_N" in self.parameter, "MAX_M_N is required in parameter"
        MAX_M_N = self.parameter["MAX_M_N"]
        assert MAX_M_N >= 1, "MAX_M_N should be greater than or equal to 1"

        M = self.parameter["M"] = random.randint(1, MAX_M_N)
        N = self.parameter["N"] = random.randint(1, MAX_M_N)
        grid = self.parameter["grid"] = [[random.randint(1, self.max_height) for j in range(N)] for i in range(M)]


        max_row = 0
        max_col = 0
        for i in range(M) :
            for j in range(N) :
                a = grid[i][j]
                t = M - i - 1
                cand_col = 2 * t + 4 * j + 6
                if cand_col > max_col :
                    max_col = cand_col
                cand_row = 2 * t + 3 * (a - 1) + 5
                if cand_row > max_row :
                    max_row = cand_row


        height = max_row + 1
        width = max_col + 1
        canvas = [['.' for _ in range(width)] for _ in range(height)]
        template = [
            "..+---+",
            "./   /|",
            "+---+ |",
            "|   | +",
            "|   |/.",
            "+---+.."
        ]


        for i in range(M) :
            for j in range(N) :
                a = grid[i][j]
                t = M - i - 1
                for k in range(a) :
                    x_offset = 2 * t + 4 * j
                    y_offset = 2 * t + 3 * k
                    for r in range(6) :
                        for c in range(7) :
                            ch = template[r][c]
                            if ch != '.' :
                                row_index = y_offset + (5 - r)
                                col_index = x_offset + c
                                canvas[row_index][col_index] = ch

        output_lines = []
        for row in range(height - 1, -1, -1) :
            output_lines.append("".join(canvas[row]))
        self.parameter["reference_answer"] = "\n".join(output_lines)
    
    def _prompt_generate(self) -> str :
        prompt = self.prompt_template
        prompt = prompt.replace("{M}", str(self.parameter["M"]))
        prompt = prompt.replace("{N}", str(self.parameter["N"]))
        prompt = prompt.replace("{matrix}", "\n".join(" ".join(map(str, row)) for row in self.parameter["grid"]))
        return prompt


    def _process(self, answer : Optional[str]) -> Optional[List[str]] :
        if answer is not None :
            answer = answer.strip()
            image = []
            for line in answer.splitlines() :
                line = line.strip()
                if line :
                    image.append(line)
            return image
        else :
            return None
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            image = processed_result

            if not image :
                return self.rewards["wrong_format"]
            for row in image :
                if len(row) != len(image[0]) :
                    return self.rewards["wrong_format"]
                if not all(ch in ".+-/| " for ch in row) :
                    return self.rewards["invalid_answer"]
            
            gold_image = self.parameter["reference_answer"].split("\n")
            if len(image) != len(gold_image) :
                return self.rewards["wrong_size"]
            if len(image[0]) != len(gold_image[0]) :
                return self.rewards["wrong_size"]
            
            total_correct = 0
            for gold_row, row in zip(gold_image, image) :
                assert len(gold_row) == len(row)
                total_correct += sum(gold_row[i] == row[i] for i in range(len(gold_row)))
            total_cells = len(gold_image) * len(gold_image[0])
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * (((total_correct / total_cells)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (total_correct == total_cells)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]