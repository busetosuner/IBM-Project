import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import Import_File
import Binning
import Numerical_Data
import Duplicates
import Regression

import handle_missing_values
import dummy_variables
import pca
import factor_analysis
import classification
import clustering

# Path will be given by user

df, file_path = Import_File.check_data_format()

# Get input from user
headers = df.columns.values

for i, header in enumerate(headers):
    print(i," ", header)

target = headers[int(input("Please enter index of target attribute: "))]
group_list = [headers[int(item)] for item in input("Please enter the index of attributes you want(leave blank for all): ").split()]

if len(group_list) != 0:
    if not target in group_list:
        group_list.append(target)
    df = df[group_list]

print("\nData is preparing...")
df = handle_missing_values.clean_missing(df, target)
df = Duplicates.clean_duplicates(df)

# Convert numeric columns to numeric in pandas for further operations
for column in df.columns.values:
    if not (Binning.is_numeric(df[column])):
        continue
    df[column] = pd.to_numeric(df[column])

df = Numerical_Data.drop_outliers(df)
# Normalization and standardization will be used in just necessary parts
# df = Numerical_Data.normalization(df)
# df = Numerical_Data.standardization(df)

for attribute in df.columns.values:
    if  (attribute == target):
        continue

    if (df[attribute].nunique() <= 5):
        # Pass numerical variables for sake of simplicity 
        if (Binning.is_numeric(df[attribute])):
            continue
        df = dummy_variables.create_dummies(df, attribute)
    else:
        df = Binning.make_bins(df, attribute)

print("\n New columns: ", df.columns.values)

df_numeric = df[Numerical_Data.numeric_columns(df)]

if target in df_numeric.columns:
    df_numeric = df_numeric.drop(target, axis = 1)

pca.pca_analysis(df_numeric)

df.drop(columns=df_numeric.columns, inplace= True)

df_numeric = factor_analysis.feature_selection(df_numeric, target, len(df_numeric.axes[0]))

df = pd.concat([df, df_numeric], axis = 1) 

# After selection update df_numeric
df_numeric = df[Numerical_Data.numeric_columns(df)]

model, mse, r2, df = Regression.perform_multiple_linear_regression(df_numeric, target)

if(len(df_numeric.axes[1]) < 20):
    classification.KNN(df_numeric, df[target], 3)
else:
    print("Sorry, this to much :(")

clustering.cluster(df_numeric, 3)

df.to_csv(file_path, index = False)
