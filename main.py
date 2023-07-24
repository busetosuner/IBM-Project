import pandas as pd
import numpy as np
import seaborn as sns

import Import_File
import Binning
import Numerical_Data
import Duplicates

import handle_missing_values
import dummy_variables
import pca
import factor_analysis


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

# df = Numerical_Data.drop_outliers(df)
df = Numerical_Data.normalization(df)
df = Numerical_Data.standardization(df)

n_c = Numerical_Data.numeric_columns(df)
df_numeric = df[n_c]

if target in df_numeric.columns:
    df_numeric = df_numeric.drop(target, axis = 1)
 
pca.pca_analysis(df_numeric)

df.drop(columns=df_numeric.columns, inplace= True)

print("\n",df_numeric.columns.values)

df_numeric = factor_analysis.feature_selection(df_numeric, target, len(df_numeric.axes[0]))

print("\n",df_numeric.columns.values)

df = pd.concat([df, df_numeric], axis = 1)

# Create  dictionary that keeps attribute names, while addingg dummy columns or creating bins update dictionary to get this new added columns
mp = {}

for header in headers:
    mp[header] = header


for attribute in group_list:
    if not (attribute in df.columns):
        print("\nSorry, this attribute {} is not correlated to target".format(attribute))
        continue

    if (df[attribute].nunique() <= 5):
        # Pass numerical variables for sake of simplicity 
        if (Binning.is_numeric(df[attribute])):
            continue

        df, mp[attribute] = dummy_variables.create_dummies(df, attribute)
    else:
        df, mp[attribute] = Binning.make_bins(df, attribute)
    print("\nNew columns of",attribute,":",mp[attribute])

print("\n New columns: ", df.columns.values)

# After this point we get header names using dictionary


df.to_csv(file_path + "df_new.csv", index = False)
