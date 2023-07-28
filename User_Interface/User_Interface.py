from tkinter import *
from tkinter import filedialog
import time
from tkinter import ttk
from tkinter import messagebox
import tkinter as tk
import Data_Collection.Import_File as Import_File

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




def show_data_format(data_format):
    messagebox.showinfo("The data Format you provided is: ", f"Dataset format: {data_format}")

def prepare_data(df, target):
    progress_window = Toplevel()  # Use Toplevel instead of Tk
    progress_window.title("Data Preparation Progress")
    progress_window.geometry("400x150")  # Set the size of the progress window

    # Center the progress window on the screen
    screen_width = progress_window.winfo_screenwidth()
    screen_height = progress_window.winfo_screenheight()
    x_coordinate = int((screen_width / 2) - (400 / 2))
    y_coordinate = int((screen_height / 2) - (150 / 2))
    progress_window.geometry(f"400x150+{x_coordinate}+{y_coordinate}")

    # Add a title to the progress window
    progress_label = Label(progress_window, text="Data Preparation \n Progressing ...", font=("Helvetica", 15), padx=20, pady=20)
    progress_label.pack()

    progress_bar_frame = Frame(progress_window)
    progress_bar_frame.pack(pady=20)

    # Make the progress bar larger and centered within the window
    progress_bar = ttk.Progressbar(progress_bar_frame, length=300, mode='determinate')
    progress_bar.pack()

    num_steps = 5

    def update_progress_bar(step):
        progress = int((step / num_steps) * 100)
        progress_bar['value'] = progress
        progress_window.update()

    update_progress_bar(0)

    for i in range(num_steps):
        time.sleep(1.5)  # Simulate data preparation time (replace with your actual data preparation steps)
        update_progress_bar(i + 1)

    progress_window.destroy()
