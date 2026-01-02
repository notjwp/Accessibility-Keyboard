import tkinter as tk

root = tk.Tk()
root.title('Tkinter Test Window')
root.geometry('400x200')
label = tk.Label(root, text='If you see this, Tkinter works!', font=('Arial', 16))
label.pack(pady=40)
root.mainloop()
