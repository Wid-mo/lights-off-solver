import numpy as np
from itertools import product


def print_solution(sol: np.ndarray[int, int]) -> None:
    for row in sol:
        for col in range(len(row)):
            print("X" if row[col] == 1 else ".", end="")
        print()


# DEDUCING SOLUTION FROM FIRST ROW
def get_solution_from_solved_first_row(
    row: np.ndarray[int],
    actual_state: np.ndarray[int, int],
    destination_state: np.ndarray[int, int],
) -> np.ndarray[int, int]:
    """
    Example
    -------
    >>> get_solution_from_solved_first_row(np.array([1, 0, 0, 0]),
    ... np.array([
    ... [1, 1, 0, 0],
    ... [1, 0, 0, 0],
    ... [0, 0, 0, 1],
    ... [0, 0, 1, 1]]),
    ... np.zeros((4, 4)))
    array([[1, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])
    """
    SHIFTS = ((0, -2), (-1, -1), (0, -1), (1, -1))
    rows, cols = actual_state.shape

    isInside = lambda row, col: row >= 0 and row < rows and col >= 0 and col < cols
    to_Z2 = lambda val: val % 2

    solution = np.empty(shape=(rows, cols), dtype=int)
    solution[0] = row
    for row in range(1, len(solution)):
        for col in range(len(solution[0])):
            num_of_cells_which_changed_above_cell_state = sum(
                isInside(row + dy, col + dx) and solution[row + dy, col + dx] == 1
                for dx, dy in SHIFTS
            )
            above_cell_state = to_Z2(num_of_cells_which_changed_above_cell_state)
            above_cell_state ^= actual_state[row - 1, col]
            above_cell_destination_state = destination_state[row - 1, col]
            solution[row, col] = abs(above_cell_destination_state - above_cell_state)
    return solution


def get_solutions_from_matrix_in_row_Echelon_form(
    M: np.ndarray[int, int], r_zero_num: int = 0
) -> np.ndarray[int, int]:
    # no solution (contradictory arrangement)
    if any(M[-1, M.shape[0] - r_zero_num :] == 1):
        return []

    # number of solutions depends on number of rows with zeros
    shape = (2**r_zero_num, M.shape[1] - 1)
    solutions = np.zeros(shape, dtype=int)
    solutions[:, -r_zero_num:] = tuple(product({0, 1}, repeat=r_zero_num))

    toZ2 = lambda v: v % 2
    for solution in solutions:
        for row in reversed(range(len(M) - r_zero_num)):
            solution[row] = toZ2(sum(M[row, (row + 1) : -1] * solution[(row + 1) :]))
            solution[row] ^= M[row, -1]

    return solutions


def transform_matrix_to_row_Echelon_form(
    M: np.ndarray[int, int], from_col: int = 0
) -> (np.ndarray[int, int], int):
    RIGHT = (0, 1)
    RIGHT_DOWN = (1, 1)
    r_zero_num = 1
    pivot = (from_col, from_col + 1)

    for _ in range(pivot[1], M.shape[1] - 1):
        if all(M[pivot[0], pivot[1] :] == 0):
            r_zero_num += 1
            pivot += RIGHT
            continue

        # swap rows and clear all cells below pivot
        r1 = pivot[0]
        r2 = np.where(M[r1 + 1 :, r1] == 1)[0][0] + (r1 + 1)
        M[[r1, r2]] = M[[r2, r1]]
        M[r1 + 1 :][M[r1 + 1 :, r1] == 1] ^= M[r1]

        pivot += RIGHT_DOWN

    return M, r_zero_num


def linalg_solve_Z2(A: np.matrix[int], b: np.ndarray[int]) -> np.ndarray[int, int]:
    M = np.empty(shape=(A.shape[0], A.shape[1] + 1), dtype=int)
    M[:, :-1] = A
    M[:, -1] = b

    # Gauss elimination
    for r1, _ in enumerate(M.diagonal()):
        if M[r1, r1] == 0:
            # if not unique solution
            if all(M[r1 + 1 :, r1] == 0):
                M, r_zero_num = transform_matrix_to_row_Echelon_form(M, from_col=r1)
                return get_solutions_from_matrix_in_row_Echelon_form(M, r_zero_num)

            # swap rows
            r2 = np.where(M[r1 + 1 :, r1] == 1)[0][0] + (r1 + 1)
            M[[r1, r2]] = M[[r2, r1]]

        # zeros all cells in column below r1
        M[r1 + 1 :][M[r1 + 1 :, r1] == 1] ^= M[r1]

    toZ2 = lambda v: v % 2

    # get solution from M
    solution = np.zeros(M.shape[1] - 1, dtype=int)
    for row in reversed(range(len(M))):
        solution[row] = toZ2(sum(M[row, (row + 1) : -1] * solution[(row + 1) :]))
        solution[row] ^= M[row, -1]

    return np.array(solution, ndmin=2)


# FIND MATRIX TO SOLVING LINEAR EQUATION
def get_matrix_contains_vectors_from_each_cell(
    actual_state: np.ndarray[int, int],
    destination_state: np.ndarray[int, int],
) -> np.ndarray[(int, int), int]:
    """
    Example for 3x3
    ---------------

    >>> get_matrix_contains_vectors_from_each_cell(np.array([
    ... [1, 1, 0],
    ... [1, 0, 1],
    ... [0, 1, 1],
    ... ]), np.zeros((3, 3)))
    array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
           [[1, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 0]],
           [[1, 0, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0]],
           [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 1]],
    ])

    For each column from second row to end, add up all columns for cell:
    above, above-above, above-left, above-right.
    For last column additionally sum difference between actual state
    and destination state for cell above.
    """
    SHIFTS = ((0, -2), (-1, -1), (0, -1), (1, -1))
    rows, cols = actual_state.shape

    isInside = lambda row, col: row >= 0 and row < rows and col >= 0 and col < cols
    to_Z2 = lambda val: val % 2

    shape = (rows + 1, cols, cols + 1)  # matrix of vectors with vectors size: cols + 1
    M = np.empty(shape=shape, dtype=int)
    M[0] = np.eye(cols, cols + 1)  # initialize first row of M
    for row in range(1, len(M)):  # len(M) is number_of_rows_in_board + 1
        for col in range(len(M[0])):
            cells_inside_board = [
                M[row + dy, col + dx]
                for dx, dy in SHIFTS
                if isInside(row + dy, col + dx)
            ]
            cell_vector = to_Z2(np.sum(cells_inside_board, axis=0))
            cell_vector[-1] ^= abs(
                destination_state[row - 1, col] - actual_state[row - 1, col]
            )
            M[row, col] = cell_vector

    return M


def solve(
    actual_state: np.ndarray[int, int], destination_state: np.ndarray[int, int]
) -> np.ndarray[np.ndarray[int, int]]:
    """
    Returns list of 2d solutions.

    Example:
    >>> solve(np.array([[1, 1]]), np.zeros((1, 2), dtype=int))
    array([[[1, 0]], [[0, 1]]])
    """
    virtual_row = get_matrix_contains_vectors_from_each_cell(
        actual_state, destination_state
    )[
        -1
    ]  # extract virtual row from M

    # # SOLVE SYSTEM OF LINEAR EQUATION IN Z2
    A, b = virtual_row[:, :-1], virtual_row[:, -1]
    x = linalg_solve_Z2(A, b)

    solutions = np.array(
        [
            get_solution_from_solved_first_row(sol_vec, actual_state, destination_state)
            for sol_vec in x
        ]
    )
    return solutions


def main() -> None:
    actual_state = np.ones((4, 4), dtype=int)
    destination_state = np.zeros(actual_state.shape, dtype=int)
    solutions = solve(actual_state, destination_state)

    print(min(solutions, key=lambda solution: np.count_nonzero(solution)))
    print(solutions)


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------------
# TESTING
# method: get_solution_from_solved_first_row
np.testing.assert_array_almost_equal(
    get_solution_from_solved_first_row(
        np.array([1, 0, 0, 0]),
        np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]]),
        np.zeros((4, 4), dtype=int),
    ),
    np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),
)
# solve for more complicated example
np.testing.assert_array_equal(
    solve(np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]]), np.zeros((3, 3), dtype=int)),
    np.array([[[0, 1, 1], [0, 1, 0], [0, 0, 1]]]),
)
# solve for one solution
np.testing.assert_array_equal(
    solve(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), np.zeros((3, 3), dtype=int)),
    np.array([[[1, 0, 1], [0, 1, 0], [1, 0, 1]]]),
)
# solve for many solutions
np.testing.assert_array_equal(
    solve(np.array([[1, 1]]), np.zeros((1, 2), dtype=int)),
    np.array([[[1, 0]], [[0, 1]]]),
)
# solve for no solution
np.testing.assert_array_equal(
    solve(np.array([[1, 0]]), np.zeros((1, 2), dtype=int)), []
)
# 4x4 where each first row combination is a solution
np.testing.assert_equal(
    len(solve(np.ones((4, 4), dtype=int), np.zeros((4, 4), dtype=int))), 16
)
