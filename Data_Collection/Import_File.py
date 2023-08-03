import os
import pandas as pd
import sys
import tkinter as tk

from tkinter import messagebox
from tkinter import * 
from tkinter import filedialog

import User_Interface.User_Interface as User_Interface

def browse_file(root):
    
    root.withdraw()  # Hide the main tkinter window

    file_path = filedialog.askopenfilename()
    # root.destroy()  # Destroy the extra window after file selection
    print(file_path)
    return file_path

def check_data_format():
    root = tk.Tk()
    User_Interface.welcome_message(root)  # Display welcome message
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

    User_Interface.show_data_format(data_format)
    tempTuple = os.path.splitext(file_path)
    file_path = tempTuple[0] + "_new" + file_extension 

    print(file_path)

    df.to_csv(file_path, index=False)
    df = pd.read_csv(file_path, index_col=False)

    return df, file_path
