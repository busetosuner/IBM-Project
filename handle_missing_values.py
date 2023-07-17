import pandas as pd
import numpy as np

# Import Data Frame

file_path = r"C:\Users\hacer\OneDrive\Masaüstü\IBM\datasets\ds_salaries.csv"

df = pd.read_csv(file_path)

# Get headers
headers = df.columns.values

# Identify missing values -assuming missing values represented '?' symbol in dataset-
df.replace("?", np.nan, inplace = True)


# Count missing values per column
missing_data = df.isnull()
missing_data_counts = []

for column in missing_data.columns.values.tolist():
    missing_data_counts.append(missing_data[column].value_counts())
    
"""
 Handling missing data methods  
    - Remove the missing data - entire row-
    - Retain data missing
    - Filling missing values previous one, next one
    - Replace with mean/mode/median
    - Replace it by frequency

 To achieve more reliable results, the following methods will be applied in this project:
    - Missing numeric values will be replaced with mean value in the column
    - Missing strings will be replaces with most frequencist value in the column
    - If the missing values is in the target column, then the row will be dropped
"""

# Missing data in target column

# target will be determined by the user, in this case it is just random
target = headers[2]

# the original dataset will be protected, ?we can create new dataset file for other steps?
df_new = df.dropna(subset=[target], axis=0)

# Missing strings and numeric values

def is_numeric(col):
    try:
        pd.to_numeric(col)
        return True
    except:
        return False

# because of creating copy of dataframe, prevent the chained assignment error 
pd.options.mode.chained_assignment = None

for header in headers:
    if is_numeric(df[header]) :
        avg = df_new[header].astype('float').mean(axis=0) 

        avg = int(avg) # year, age, model cannot be float
        df_new[header].replace(np.nan, avg, inplace= True)
    else:
        most_common = df_new[header].value_counts().idxmax()
        df_new[header].replace(np.nan, most_common, inplace=True)
   