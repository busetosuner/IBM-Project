import pandas as pd
import numpy as np

"""
To achieve more reliable results, the following methods will be applied in this project:
    - Missing numeric values will be replaced with mean value in the column
    - Missing strings will be replaces with most frequencist value in the column
    - If the missing values is in the target column, then the row will be dropped
"""

def clean_missing(df, target):
    # Get headers
    headers = df.columns.values

    # Identify missing values -assuming missing values represented '?' symbol in dataset-
    df.replace("?", np.nan, inplace = True)

    # Count missing values per column
    missing_data = df.isnull()
    missing_data_counts = []

    for column in missing_data.columns.values.tolist():
        missing_data_counts.append(missing_data[column].value_counts())

    avg = (df.isnull().sum()).sum()/(len(df.axes[0])*len(df.axes[1])*100)

    print("The missing value percentage: %{:.3f}".format(avg))
    # Missing data in target column
    df.dropna(subset=[target], axis=0, inplace=True)

    # Missing strings and numeric values
    def is_numeric(col):
        try:
            pd.to_numeric(col)
            return True
        except:
            return False

    for header in headers:
        if is_numeric(df[header]) :
            avg = df[header].astype('float').mean(axis=0) 

            avg = int(avg) # year, age, model cannot be float
            df[header].replace(np.nan, avg, inplace= True)
        else:
            most_common = df[header].value_counts().idxmax()
            df[header].replace(np.nan, most_common, inplace=True)

    return df
