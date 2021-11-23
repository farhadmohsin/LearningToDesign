import tkinter as tk

window = tk.Tk()

txt1 = tk.Label(text = 'Input', font='Helvetica 18 bold')
txt1.pack()

frame1 = tk.Frame()
lbl_grp = tk.Label(text = 'Group Sizes (comma separated)', master = frame1, width=30, anchor='w')
entry_grp = tk.Entry(master=frame1)
lbl_grp.grid(row = 1, column = 1)
entry_grp.grid(row = 1, column = 2)
frame1.pack()

frame2 = tk.Frame()
lbl_m = tk.Label(text = 'No. of candidates', master = frame2, width=30, anchor='w')
entry_m = tk.Entry(master=frame2)
lbl_m.grid(row = 1, column = 1)
entry_m.grid(row = 1, column = 2)
frame2.pack()

frame3 = tk.Frame()
lbl_cond = tk.Label(text = 'Condorcet efficiency required', master = frame3, width=30, anchor='w')
entry_cond = tk.Entry(master=frame3)
lbl_cond.grid(row = 1, column = 1)
entry_cond.grid(row = 1, column = 2)
frame3.pack()

frame4 = tk.Frame()
lbl_fair = tk.Label(text = 'Group fairness required', master = frame4, width=30, anchor='w')
entry_fair = tk.Entry(master=frame4)
lbl_fair.grid(row = 1, column = 1)
entry_fair.grid(row = 1, column = 2)
frame4.pack()

frame5 = tk.Frame()
lbl_data = tk.Label(text = 'Simulated data size', master = frame5, width=30, anchor='w')
entry_data = tk.Entry(master=frame5)
lbl_data.grid(row = 1, column = 1)
entry_data.grid(row = 1, column = 2)
frame5.pack()

btn_compute = tk.Button(text = 'Find Voting Rule', font='Helvetica 12 bold')
btn_compute.pack()

txt2 = tk.Label(text = 'Output', font='Helvetica 18 bold')
txt2.pack()

text_box = tk.Text(width = 100, height = 10)
text_box.pack()
# text_box.insert("1.0", '''No existing or newly designed voting rules empirically satisfy the requirements
# Copeland - Condorcet Efficiency = 1.0, Group Fairness = 0.89
# Maximin - Condorcet Efficiency = 1.0, Group Fairness = 0.89
# Borda - Condorcet Efficiency = 0.94, Group Fairness = 0.89
# STV - Condorcet Efficiency = 0.95, Group Fairness = 0.89

# New generated voting rules 0.6-ML, 0.5-ML
# 0.6-ML - Condorcet Efficiency = 0.71, Group Fairness = 0.95
# 0.7-ML - Condorcet Efficiency = 0.79, Group Fairness = 0.92''')

text_box.insert("1.0", '''Existing voting rules Copeland and Maximin empirically satisfy the requirements
Copeland - Condorcet Efficiency = 1.0, Group Fairness = 0.89
Maximin - Condorcet Efficiency = 1.0, Group Fairness = 0.89
''')


btn_upload = tk.Button(text = 'Upload new preference profile', font='Helvetica 12 bold')
btn_upload.pack()

btn_upload = tk.Button(text = 'Choose Voting Rule to Use', font='Helvetica 12 bold')
btn_upload.pack()

# frame = tk.Frame()

# category = tk.Label(text = 'Categories', font='Helvetica 18 bold', master=frame)

# temp = tk.Label(text = 'Left')
# in_a = tk.Entry(master=frame)
# in_b = tk.Entry()

# output = tk.Entry(text = 'Output')

# category.grid(row = 1, column = 1)
# in_a.grid(row = 1, column = 2)

# frame.pack()

# in_b.pack()

# def get():
#     a = in_a.get()
#     b = in_b.get()
#     output.delete(0, tk.END)
#     output.insert(0, str(int(a)+int(b)))

# button = tk.Button(text ="Button", command = get)
# button.pack()
    
# output.pack()

# text_box = tk.Text(width = 50, height = 5)
# text_box.pack()

window.mainloop()
