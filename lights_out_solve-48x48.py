import numpy as np
from functools import reduce
from typing import Callable


def pipe(*func: Callable) -> Callable:
    def pipe(f, g):
        return lambda x: g(f(x))

    return reduce(pipe, func, lambda x: x)


# -------------------------------------


def print_solution(sol: np.ndarray[int, int]) -> None:
    for row in sol:
        for col in range(len(row)):
            print("X" if row[col] == 1 else " ", end="")
        print()


# DEDUCING SOLUTION FROM FIRST ROW
def get_solution_from_solved_first_row(
    row: np.ndarray[int], rows: int, cols: int
) -> np.ndarray[int, int]:
    SHIFTS = ((0, -2), (-1, -1), (0, -1), (1, -1))
    isInside = lambda row, col: row >= 0 and row < rows and col >= 0 and col < cols

    get_number_of_cells_influencing_cell_above = lambda cells: sum(cells)
    to_Z2 = lambda val: val % 2
    negate = lambda val: 1 - val

    solution = np.empty(shape=(rows, cols), dtype=int)
    solution[0] = row
    for row in range(1, len(solution)):
        for col in range(len(solution[0])):
            cells_which_influence_cell_above_this_cell = (
                solution[row + dy][col + dx]
                for dx, dy in SHIFTS
                if isInside(row + dy, col + dx) and solution[row + dy][col + dx] == 1
            )

            solution[row][col] = pipe(
                get_number_of_cells_influencing_cell_above, to_Z2, negate
            )(cells_which_influence_cell_above_this_cell)
    return solution


def linalg_solve_Z2(A: np.matrix[int], b: np.ndarray[int]) -> np.ndarray[int]:
    M = np.empty(shape=(A.shape[0], A.shape[1] + 1), dtype=int)
    M[:, :-1] = A
    M[:, -1] = b

    # Gauss elimination
    for r1, _ in enumerate(M.diagonal()):
        if M[r1][r1] == 0:
            no_cell_with_ones = lambda list: sum(list) == 0

            if no_cell_with_ones(M[r1 + 1 :, r1]):
                raise ValueError("No Unit Solution Exception")
            # swap rows
            r2 = np.where(M[r1 + 1 :, r1] == 1)[0][0] + (r1 + 1)
            M[[r1, r2]] = M[[r2, r1]]

        # zeros all cells in column below r1
        M[r1 + 1 :][M[r1 + 1 :, r1] == 1] ^= M[r1]

    # get solution from M
    solution = np.zeros(M.shape[1] - 1, dtype=int)
    toZ2 = lambda v: v % 2
    for row in reversed(np.arange(len(M))):
        solution[row] = toZ2(sum(M[row, (row + 1) : -1] * solution[(row + 1) :]))
        solution[row] ^= M[row, -1]

    return solution


# FIND MATRIX TO SOLVING LINEAR EQUATION
def get_matrix_contains_vectors_from_each_cell(
    rows: int, cols: int
) -> np.ndarray[(int, int), int]:
    SHIFTS = ((0, -2), (-1, -1), (0, -1), (1, -1))
    isInside = lambda row, col: row >= 0 and row < rows and col >= 0 and col < cols

    get_number_of_cells_influencing_cell_above = lambda cells: np.sum(
        list(cells), axis=0
    )
    to_Z2 = lambda val: val % 2
    negate = lambda vec: np.array(list(vec[:-1]) + [1 - vec[-1]])

    shape = (rows + 1, cols, cols + 1)  # matrix of vectors with vectors size: cols + 1
    M = np.empty(shape=shape, dtype=int)
    M[0] = np.eye(rows, rows + 1)  # initialize first row of M
    for row in range(1, len(M)):  # len(M) is number_of_rows_in_board + 1
        for col in range(len(M[0])):
            cells_inside_board = (
                M[row + dy][col + dx]
                for dx, dy in SHIFTS
                if isInside(row + dy, col + dx)
            )
            M[row][col] = pipe(
                get_number_of_cells_influencing_cell_above, to_Z2, negate
            )(cells_inside_board)

    return M


def main() -> None:
    rows, cols = 48, 48
    virtual_row = get_matrix_contains_vectors_from_each_cell(rows, cols)[
        -1
    ]  # extract virtual row from M

    # SOLVE SYSTEM OF LINEAR EQUATION IN Z2
    A, b = virtual_row[:, :-1], virtual_row[:, -1]
    x = linalg_solve_Z2(A, b)

    SOLUTION = get_solution_from_solved_first_row(x, rows, cols)
    print_solution(SOLUTION)


if __name__ == "__main__":
    main()
