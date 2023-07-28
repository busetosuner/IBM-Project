from tkinter import *
from tkinter import filedialog
import time
from tkinter import ttk
from tkinter import messagebox
import tkinter as tk


    
def get_file_path(file_path):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path


# Function to display the welcome message
def welcome_message(root):
    root.withdraw()  # Hide the main tkinter window
    welcome_msg = "Welcome to Scourgify! Please select a dataset to proceed."
    messagebox.showinfo("Welcome", welcome_msg)



def prepare_data(df, target):
    progress_window = Tk()
    progress_window.title("Data Preparation Progress")
    progress_label = Label(progress_window, text="Data Preparation Progress", padx=20, pady=20)
    progress_label.pack()
    progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
    progress_bar.pack()

    num_steps = 5

    def update_progress_bar(step):
        progress = int((step / num_steps) * 100)
        progress_bar['value'] = progress
        progress_window.update()

    update_progress_bar(0)

    for i in range(num_steps):
        time.sleep(1)  # Simulate data preparation time (replace with your actual data preparation steps)
        update_progress_bar(i + 1)

    progress_window.destroy()
