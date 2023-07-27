import pandas as pd
import warnings
from sklearn.preprocessing import OrdinalEncoder
warnings.filterwarnings("ignore")

import Import_File
import Binning
import Numerical_Data
import Duplicates
import Regression
import Correlation
#import clustering2 

import handle_missing_values
import dummy_variables
import classification
import clustering
import feature_selection

from tqdm import tqdm  # Import tqdm for progress bar
import time  # Import time to simulate loading time
from tkinter import *
from tkinter import ttk

warnings.filterwarnings("ignore")

# Import the necessary modules and functions (your import statements)

# Path will be given by the user
df, file_path = Import_File.check_data_format()

# Get input from the user for the target attribute and the group list
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

# Get input from the user for the target attribute and the group list
headers = df.columns.values

for i, header in enumerate(headers):
    print(i, " ", header)

target = headers[int(input("Please enter index of target attribute: "))]
group_list_input = input("Please enter the index of attributes you want(leave blank for all): ").strip()
group_list = [headers[int(item)] for item in group_list_input.split()] if group_list_input else []

if len(group_list) != 0:
    if not target in group_list:
        group_list.append(target)
    df = df[group_list]
elif "Unnamed: 0" in df.columns.values:
    df = df.drop("Unnamed: 0", axis=1)

prepare_data(df, target)  # Call the function here to display the progress window

# Clean missing values and duplicates
df = handle_missing_values.clean_missing(df, target)
df = Duplicates.clean_duplicates(df)


# Convert numeric columns to numeric in pandas for further operations
for column in df.columns.values:
    if not Binning.is_numeric(df[column]):
        continue
    df[column] = pd.to_numeric(df[column])

# Data preprocessing steps
df = Numerical_Data.drop_outliers(df)
df = Numerical_Data.normalization(df)
df = Numerical_Data.standardization(df)

for attribute in df.columns.values:
    if attribute == target:
        if not Binning.is_numeric(df[target]):
            encoder = OrdinalEncoder()
            df[target] = encoder.fit_transform(df[[target]])
        continue

    if df[attribute].nunique() <= 5:
        # Pass numerical variables for the sake of simplicity
        if Binning.is_numeric(df[attribute]):
            continue
        df = dummy_variables.create_dummies(df, attribute)
    else:
        df = Binning.make_bins(df, attribute)

print("\n New columns: ", df.columns.values)

df = feature_selection.decide(df, target)

df_numeric = df[Numerical_Data.numeric_columns(df)]

target_correlation = Correlation.calculate_correlation(df, target)

model, mse, r2, df = Regression.perform_multiple_linear_regression(df_numeric, target)

if len(df_numeric.axes[1]) < 20:
    classification.KNN(df_numeric, target, 3)
else:
    print("Sorry, this is too much for KNN classification :(")

classification.decision_trees(df, target)

#clustering2.cluster(df_numeric, 3)

df.to_csv(file_path, index=False)
