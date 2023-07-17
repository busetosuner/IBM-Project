import pandas as pd
import numpy as np

# Import Data Frame

file_path = r"C:\Users\hacer\OneDrive\Masaüstü\IBM\datasets\df.csv"

df = pd.read_csv(file_path)

# Get headers
headers = df.columns.values

# Dummy variables 

# Keep original file, create new one
df_new = df

pd.options.mode.chained_assignment = None

# for each column if there are equal or less than 3 unşque values turn into dummy variables
for header in headers:
    if (df[header].nunique() <= 3):

        # Create new dataframe for dummy variables
        dummy_variable = pd.get_dummies(df[header], prefix=header)

        # Turn True/False into 1/0 
        dummy_variable = dummy_variable.astype(int)

        # Add to main dataframe
        df_new = pd.concat([df_new, dummy_variable], axis=1)

        # Drop the original column
        df_new.drop(header, axis=1, inplace=True)

# Export the new file -this lines will be deleted after other steps is completed-

path = 'C:\\Users\\hacer\\OneDrive\\Masaüstü\\IBM\\datasets\\'
df_new.to_csv(path + "df.csv", index=False)