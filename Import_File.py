from tkinter import messagebox
import os
import pandas as pd
import sys

from tkinter import * 
from tkinter import filedialog
import tkinter as tk

# Function to display the welcome message
def welcome_message(root):
    root.withdraw()  # Hide the main tkinter window
    welcome_msg = "Welcome to Scourgify! Please select a dataset to proceed."
    messagebox.showinfo("Welcome", welcome_msg)

def browse_file(root):
    
    root.withdraw()  # Hide the main tkinter window

    file_path = filedialog.askopenfilename()
    # root.destroy()  # Destroy the extra window after file selection
    print(file_path)
    return file_path



# Hangi data formatında kontrol ediyor
def check_data_format():
    root = tk.Tk()
    welcome_message(root)  # Display welcome message
    file_path = browse_file(root)
    _, file_extension = os.path.splitext(file_path)
    
    # Creating empty dataFrame
    df = pd.DataFrame()

    data_format = ""
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
        data_format = 'csv'
    elif file_extension == '.xlsx' or file_extension == '.xls':
        df = pd.read_excel(file_path)
        data_format = 'excel'
    elif file_extension == '.json':
        df = pd.read_json(file_path)
        data_format = 'json'
    else:
        print("Error")
        sys.exit()

    print("Dataset:", data_format)
    tempTuple = os.path.splitext(file_path)
    file_path = tempTuple[0] + "_new" + file_extension 

    print(file_path)

    df.to_csv(file_path, index=False)
    df = pd.read_csv(file_path, index_col=False)

    return df, file_path
