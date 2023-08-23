from tkinter import *
from tkinter import ttk
from numpy import zeros, array, count_nonzero
from lights_out_solve import solve
from operator import add
from itertools import product


window = Tk()
window.title("Lights off solver")
window.geometry("{}x{}".format(45 * 9, 45 * 16))

# top toolbar
toolbar_frm = Frame(window)
toolbar_frm.pack(fill="x")
Button(toolbar_frm, text="reset").pack(side="left", fill="both", expand=True, ipady=10)
Button(toolbar_frm, text="fill all cells").pack(
    side="left", fill="both", expand=True, ipady=10
)
Button(toolbar_frm, text="random").pack(side="left", fill="both", expand=True, ipady=10)
board_type = StringVar(toolbar_frm, "3x3")
OptionMenu(
    toolbar_frm, board_type, *"1x5 2x2 3x3 4x4 5x5 6x6 7x7 8x8 48x48 nxn nxm".split()
).pack(side="left", fill="both", expand=True, padx=10)

# cells
width_num, height_num = 3, 3
cells_frm = Frame(window)
cells_frm.pack(expand=True, fill="both")
for row in range(height_num):
    cells_frm.rowconfigure(row, weight=1)
for col in range(width_num):
    cells_frm.columnconfigure(col, weight=1)
cells = [
    [Button(cells_frm, text=" ", bg="lightgray") for _ in range(height_num)]
    for _ in range(width_num)
]
for row, row_cells in enumerate(cells):
    for col, cell in enumerate(row_cells):
        cell.configure(
            command=lambda btn=cell: btn.configure(
                bg="yellow" if btn.cget("bg") == "lightgray" else "lightgray"
            )
        )
        cell.grid(row=row, column=col, sticky="nswe")


def toggle_app_mode():
    # change Button text to "Editor"
    toogle_game_mode_btn.configure(text="Editor")
    # change behaviour of cells buttons
    SHIFTS = (0, 1), (1, 0), (-1, 0), (0, -1)
    is_inside_board = (
        lambda board, row, col: row >= 0
        and col >= 0
        and row < len(board)
        and col < len(board[0])
    )
    for row, row_cells in enumerate(cells):
        for col, cell in enumerate(row_cells):
            # add_tuples = lambda t1,t2: (t1[0]+t2[0], t1[1]+t2[1])
            # neightbours_cell_indexes = (add_tuples(t1, t2) for t1,t2 in product([(row, col)], SHIFTS))
            # # neightbours_cell_indexes = tuple(
            # #     tuple(map(add, t1, t2)) for t1, t2 in product([(row, col)], SHIFTS)
            # # )  # (row, col) + SHIFTS

            # print(neightbours_cell_indexes) if row == 1 and col == 2 else ...
            # neightbours_cell_indexes2 = tuple((row_, col_) for row_, col_ in neightbours_cell_indexes if is_inside_board(cells, row_, col_))
            # print(neightbours_cell_indexes2) if row == 1 and col == 2 else ...

            # # neightbours_cells = (cells[row_][col_] for row_, col_ in neightbours_cell_indexes if is_inside_board(cells, row_, col_))
            # # neightbours_cells = tuple(cells[row_][col_] for row_, col_ in neightbours_cell_indexes2)
            # print(neightbours_cells) if row == 1 and col == 2 else ...
            # neightbours_cells = (cells[a][b] for a,b in neightbours_cell_indexes2)



            neightbours_cell_indexes = [(row+t[0], col+t[1]) for t in SHIFTS]
            neightbours_cell_indexes2 = [t for t in neightbours_cell_indexes if is_inside_board(cells, t[0], t[1])]
            # print(tuple(neightbours_cell_indexes2)) if row == 1 and col == 2 else ...

            neightbours_cells = (cells[t[0]][t[1]] for t in neightbours_cell_indexes2)
            print(tuple(neightbours_cells)) if row == 1 and col == 2 else ...


            cell.configure(
                command=lambda btn=cell: [
                    c.configure(
                        bg="yellow"
                        # bg="yellow" if c.cget("bg") == "lightgray" else "lightgray",
                        # text=c.cget("text") if c is not btn
                        # else "⭕" if btn.cget("text") == " " else " ",
                    )
                    for c in neightbours_cells
                ]
                # command=lambda btn=cell: btn.configure(
                #     bg="yellow" if btn.cget("bg") == "lightgray" else "lightgray"
                # )
                # command=lambda btn=cell: cells[0][0].configure(bg="yellow")
                # command=lambda btn=cell: [c.configure(bg="yellow") for c in neightbours_cells]
            )
    # hide 3 Buttons and show 1 button


def mark_cells():
    cells_values = [
        [1 if cell.cget("bg") == "yellow" else 0 for cell in cell_row]
        for cell_row in cells
    ]

    current_cells = array(cells_values, dtype=int)
    dest_cells = zeros((height_num, width_num), dtype=int)

    solutions = solve(current_cells, dest_cells)
    solution = min(solutions, key=lambda solution: count_nonzero(solution))

    for row, row_cells in enumerate(cells):
        for col, cell in enumerate(row_cells):
            cell.configure(text="⭕" if solution[row, col] == 1 else " ")

    toggle_app_mode()


# 2 very important buttons
execute_frm = Frame(window)
execute_frm.pack(side="bottom", fill="x")
toogle_game_mode_btn = Button(execute_frm, text="Play", command=toggle_app_mode)
toogle_game_mode_btn.pack(side="left", fill="both", expand=True, ipady=50)
Button(execute_frm, text="SOLVE", command=mark_cells).pack(
    side="left", fill="both", expand=True, ipady=50
)


# # 2 app mode
# frame = Frame(window)
# frame.pack(pady=10)
# game_mode = StringVar(frame, "Editor Mode")
# Radiobutton(
#     frame, text="Editor Mode", value="Editor Mode", variable=game_mode, indicatoron=0
# ).pack(side="left")
# Radiobutton(
#     frame, text="Game Mode", value="Game Mode", variable=game_mode, indicatoron=False
# ).pack(side="left")

# ttk.Separator(window).pack(fill="both")

# frame1 = Frame(window)
# frame1.pack(padx=10, pady=10)
# Button(frame1, text="reset").pack(side="left", padx=5)
# Button(frame1, text="fill all cells").pack(side="left", padx=5)
# Button(frame1, text="random").pack(side="left", padx=5)
# board_type = StringVar(frame1, "3x3")
# OptionMenu(frame1, board_type, *"1x5 2x2 3x3 4x4 5x5 6x6 7x7 8x8 48x48".split()).pack(
#     side="left", padx=30
# )

# ttk.Separator(window).pack(fill="both")

# # 2 buttons and slider
# change_width_frm = Frame(window)
# change_width_frm.pack(padx=10, pady=10, fill="x")
# Button(
#     change_width_frm, text="-", command=lambda: width_num.set(width_num.get() - 1)
# ).pack(side="left", expand=True, ipadx=10, ipady=10)
# width_num = IntVar(change_width_frm, 3)
# Scale(change_width_frm, from_=1, to=48, orient="horizontal", variable=width_num).pack(
#     side="left", expand=True
# )
# Button(
#     change_width_frm, text="+", command=lambda: width_num.set(width_num.get() + 1)
# ).pack(side="left", expand=True, ipadx=10, ipady=10)

# # THE MOST IMPORTANT BUTTON
# Button(window, text="SOLVE").pack(side="bottom", pady=50, ipadx=50, ipady=20)

# # vertical 2 buttons and slider
# change_height_frm = Frame(window)
# change_height_frm.pack(side="left", padx=10, pady=10, fill="y")
# Button(
#     change_height_frm, text="-", command=lambda: width_num.set(width_num.get() - 1)
# ).pack(expand=True, ipadx=10, ipady=10)
# width_num = IntVar(change_height_frm, 3)
# Scale(change_height_frm, from_=1, to=48, orient="vertical", variable=width_num).pack(
#     expand=True
# )
# Button(
#     change_height_frm, text="+", command=lambda: width_num.set(width_num.get() + 1)
# ).pack(expand=True, ipadx=10, ipady=10)

# cells_frm = Frame(window)
# cells_frm.pack(expand=True, fill="both")
# cells_frm.rowconfigure(0, weight=1)
# cells_frm.rowconfigure(1, weight=1)
# cells_frm.rowconfigure(2, weight=1)
# cells_frm.columnconfigure(0, weight=1)
# cells_frm.columnconfigure(1, weight=1)
# cells_frm.columnconfigure(2, weight=1)
# Button(cells_frm, text="⭕", bg="yellow").grid(row=0, column=0, sticky="nswe")
# Button(cells_frm, text="⭕", bg="yellow").grid(row=0, column=1, sticky="nswe")
# Button(cells_frm, text=" ", bg="yellow").grid(row=0, column=2, sticky="nswe")
# Button(cells_frm, text=" ", bg="yellow").grid(row=1, column=0, sticky="nswe")
# Button(cells_frm, text=" ", bg="lightgray").grid(row=1, column=1, sticky="nswe")
# Button(cells_frm, text=" ", bg="yellow").grid(row=1, column=2, sticky="nswe")
# Button(cells_frm, text="⭕", bg="yellow").grid(row=2, column=0, sticky="nswe")
# Button(cells_frm, text="⭕", bg="yellow").grid(row=2, column=1, sticky="nswe")
# Button(cells_frm, text="⭕", bg="yellow").grid(row=2, column=2, sticky="nswe")

window.mainloop()
