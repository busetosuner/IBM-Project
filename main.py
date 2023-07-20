import pandas as pd
import Import_File
import handle_missing_values
import Duplicates
import numpy as np
import dummy_variables
import Binning
import Numerical_Data

# Path will be given by user
file_path = 'C:\\Users\\hacer\\OneDrive\\Masaüstü\\IBM\\datasets\\'

df = Import_File.check_data_format(file_path + "ds_salaries.csv")

df.to_csv(file_path + "df_new.csv", index = False)

df = pd.read_csv(file_path + "df_new.csv", index_col = False)

# Get input from user
headers = df.columns.values

for i, header in enumerate(headers):
    print(i," ", header)

# Inputs: target = 7, attributes = 2 8 11
target = headers[int(input("Please enter index of target attribute: "))]
group_list = [headers[int(item)] for item in input("Please enter the index of attributes you want: ").split()]

df = handle_missing_values.clean_missing(df, target)

df = Duplicates.clean_duplicates(df)
print("DF: \n",df.head())

df = Numerical_Data.drop_outliers(df)
df = Numerical_Data.normalization(df)
df = Numerical_Data.standardization(df)

print("DF: \n",df.head())
# Create  dictionary that keeps attribute names, while addingg dummy columns or creating bins update dictionary to get this new added columns
mp = {}

for header in headers:
    mp[header] = header

# After this point we get header names using dictionary
 
for attribute in group_list:
    if (df[attribute].nunique() <= 5):
        # Pass numerical variables for sake of simplicity 
        if (Binning.is_numeric(df[attribute])):
            continue

        df, mp[attribute] = dummy_variables.create_dummies(df, attribute)
    else:
        df, mp[attribute] = Binning.make_bins(df, attribute)
    print("\nNew columns of",attribute,":",mp[attribute])

print("Check if the dictionary is working:" )
print("\n",df[mp["company_size"]].head())
print("\n New columns: ", df.columns.values)
print("\n",df.head())

df.to_csv(file_path + "df_new.csv", index = False)
