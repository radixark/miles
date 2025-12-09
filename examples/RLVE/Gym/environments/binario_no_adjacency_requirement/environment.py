import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Binario_NoAdjacencyRequirement_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a (2 × {N}) × (2 × {M}) matrix. Each cell contains either '0', '1', or '*' ('*' means the cell is empty). Please fill all '*' cells with either '0' or '1' such that:
1. Each **row** contains exactly {M} '0's and {M} '1's.
2. Each **column** contains exactly {N} '0's and {N} '1's.

The matrix is given in **row-major order**, with each row represented as a string of '0', '1', and '*':
{matrix}

**Output Format:** Output (2 × {N}) lines, each containing (2 × {M}) characters, where each character is either '0' or '1'. The output should match the format of the input (i.e., one row per line, no separators)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, wrong_solution : float = 0.0, correct_solution : float = 1.0,
                 **kwargs) :
        """
        Initialize the Binario_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "wrong_solution" : wrong_solution,
            "correct_solution" : correct_solution,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)

        row_permutation, col_permutation = list(range(2 * N)), list(range(2 * M))
        random.shuffle(row_permutation)
        random.shuffle(col_permutation)
        
        matrix = [[str((row_permutation[i] + col_permutation[j]) % 2) for j in range(2 * M)] for i in range(2 * N)]
        self.parameter["reference_answer"] = "\n".join("".join(row) for row in matrix)

        assert "sparsity" in self.parameter, "sparsity is required in parameter"
        sparsity = self.parameter["sparsity"]
        assert 0 < sparsity < 1, "sparsity should be between 0 and 1"
        empty_cells = random.sample(range((2 * N) * (2 * M)), max(1, int((2 * N) * (2 * M) * sparsity)))
        for cell in empty_cells :
            row, column = divmod(cell, 2 * M)
            matrix[row][column] = '*'
        self.parameter["matrix"] = ["".join(row) for row in matrix]
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            matrix = "\n".join("".join(map(str, row)) for row in self.parameter["matrix"]),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                matrix = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        matrix.append(line.strip())
                return matrix
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            N, M = self.parameter["N"], self.parameter["M"]
            solution = processed_result
            
            if len(solution) != 2 * N or any(len(row) != 2 * M for row in solution) :
                return self.rewards["wrong_format"]
            for row in solution :
                if not all(c in "01" for c in row) :
                    return self.rewards["wrong_format"]
            
            for row, original_row in zip(solution, self.parameter["matrix"]) :
                for cell, original_cell in zip(row, original_row) :
                    if original_cell != '*' and cell != original_cell :
                        assert (original_cell == '0' and cell == '1') or (original_cell == '1' and cell == '0')
                        return self.rewards["invalid_solution"]
            
            for i in range(2 * N) :
                if solution[i].count('1') != solution[i].count('0') :
                    return self.rewards["wrong_solution"]
                assert solution[i].count('1') == M, "Row {} does not have exactly {} ones".format(i, M)
                assert solution[i].count('0') == M, "Row {} does not have exactly {} zeros".format(i, M)
            for j in range(2 * M) :
                if sum(solution[i][j] == '1' for i in range(2 * N)) != sum(solution[i][j] == '0' for i in range(2 * N)) :
                    return self.rewards["wrong_solution"]
                assert sum(solution[i][j] == '1' for i in range(2 * N)) == N, "Column {} does not have exactly {} ones".format(j, N)
                assert sum(solution[i][j] == '0' for i in range(2 * N)) == N, "Column {} does not have exactly {} zeros".format(j, N)
            
            return self.rewards["correct_solution"]
        else :
            return self.rewards["wrong_format"]