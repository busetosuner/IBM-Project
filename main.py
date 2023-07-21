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


# Create  dictionary that keeps attribute names, while addingg dummy columns or creating bins update dictionary to get this new added columns
mp = {}

for header in headers:
    mp[header] = header

# After this point we get header names using dictionary
n_c = Numerical_Data.numeric_columns(df)
df_t = df[n_c]

pca.pca_analysis(df_t)

df.drop(columns=df_t.columns, inplace= True)

df_t = factor_analysis.feature_selection(df_t, target)

df = pd.concat([df, df_t], axis = 1)


print("\n",df_t.head())


for attribute in group_list:
    if not (attribute in df.columns):
        print("Sorry, this attribute {} is not correlated to target".format(attribute))

    if (df[attribute].nunique() <= 5):
        # Pass numerical variables for sake of simplicity 
        if (Binning.is_numeric(df[attribute])):
            continue

        df, mp[attribute] = dummy_variables.create_dummies(df, attribute)
    else:
        df, mp[attribute] = Binning.make_bins(df, attribute)
    print("\nNew columns of",attribute,":",mp[attribute])

print("\n New columns: ", df.columns.values)


df.to_csv(file_path + "df_new.csv", index = False)

#Regression
model, mse, r2, df = Regression.perform_multiple_linear_regression(df, target, group_list, mp)

# Print regression results
print("\nMultiple Linear Regression Results:")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)






