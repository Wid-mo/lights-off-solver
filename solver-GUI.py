from tkinter import *
from tkinter import ttk

window = Tk()
window.title("Lights off solver")
window.geometry("500x700")

# 2 app mode
frame = Frame(window)
frame.pack(pady=10)
game_mode = StringVar(frame, "Editor Mode")
Radiobutton(
    frame, text="Editor Mode", value="Editor Mode", variable=game_mode, indicatoron=0
).pack(side="left")
Radiobutton(
    frame, text="Game Mode", value="Game Mode", variable=game_mode, indicatoron=False
).pack(side="left")

ttk.Separator(window).pack(fill="both")

frame1 = Frame(window)
frame1.pack(padx=10, pady=10)
Button(frame1, text="reset").pack(side="left", padx=5)
Button(frame1, text="fill all cells").pack(side="left", padx=5)
Button(frame1, text="random").pack(side="left", padx=5)
board_type = StringVar(frame1, "3x3")
OptionMenu(frame1, board_type, *"1x5 2x2 3x3 4x4 5x5 6x6 7x7 8x8 48x48".split()).pack(side="left", padx=30)

ttk.Separator(window).pack(fill="both")

# 2 buttons and slider
change_width_frm = Frame(window)
change_width_frm.pack(padx=10, pady=10, fill="x")
Button(change_width_frm, text="-", command=lambda: width_num.set(width_num.get() - 1)).pack(side="left", expand=True, ipadx=10, ipady=10)
width_num = IntVar(change_width_frm, 3)
Scale(change_width_frm, from_= 1, to=48, orient="horizontal", variable=width_num).pack(side="left", expand=True)
Button(change_width_frm, text="+", command=lambda: width_num.set(width_num.get()+1)).pack(side="left", expand=True, ipadx=10, ipady=10)

# THE MOST IMPORTANT BUTTON
Button(window, text="SOLVE").pack(side="bottom", pady=50, ipadx=50, ipady=20)

# vertical 2 buttons and slider
change_height_frm = Frame(window)
change_height_frm.pack(side="left", padx=10, pady=10, fill="y")
Button(change_height_frm, text="-", command=lambda: width_num.set(width_num.get() - 1)).pack(expand=True, ipadx=10, ipady=10)
width_num = IntVar(change_height_frm, 3)
Scale(change_height_frm, from_= 1, to=48, orient="vertical", variable=width_num).pack(expand=True)
Button(change_height_frm, text="+", command=lambda: width_num.set(width_num.get()+1)).pack(expand=True, ipadx=10, ipady=10)

cells_frm = Frame(window)
cells_frm.pack(expand=True, fill="both")
cells_frm.rowconfigure(0, weight=1)
cells_frm.rowconfigure(1, weight=1)
cells_frm.rowconfigure(2, weight=1)
cells_frm.columnconfigure(0, weight=1)
cells_frm.columnconfigure(1, weight=1)
cells_frm.columnconfigure(2, weight=1)
Button(cells_frm, text="⭕", bg="yellow").grid(row=0, column=0, sticky="nswe")
Button(cells_frm, text="⭕", bg="yellow").grid(row=0, column=1, sticky="nswe")
Button(cells_frm, text=" ", bg="yellow").grid(row=0, column=2, sticky="nswe")
Button(cells_frm, text=" ", bg="yellow").grid(row=1, column=0, sticky="nswe")
Button(cells_frm, text=" ", bg="lightgray").grid(row=1, column=1, sticky="nswe")
Button(cells_frm, text=" ", bg="yellow").grid(row=1, column=2, sticky="nswe")
Button(cells_frm, text="⭕", bg="yellow").grid(row=2, column=0, sticky="nswe")
Button(cells_frm, text="⭕", bg="yellow").grid(row=2, column=1, sticky="nswe")
Button(cells_frm, text="⭕", bg="yellow").grid(row=2, column=2, sticky="nswe")

window.mainloop()
