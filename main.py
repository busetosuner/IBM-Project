import pandas as pd
import Import_File
import handle_missing_values
import Duplicates
import numpy as np

# Path will be given by user
file_path = 'C:\\Users\\hacer\\OneDrive\\Masaüstü\\IBM\\datasets\\'

df = Import_File.check_data_format(file_path + "ds_salaries.csv")

df.to_csv(file_path + "df_new.csv", index = False)

df = pd.read_csv(file_path + "df_new.csv", index_col = False)

# Get input from user
headers = df.columns.values

for i, header in enumerate(headers):
    print(i," ", header)

target = headers[int(input("Please enter index of target attribute: "))]
group_list = [headers[int(item)] for item in input("Please enter the index of attributes you want: ").split()]

df = handle_missing_values.clean_missing(df, target)

df = Duplicates.clean_duplicates(df)

print(df.head())
