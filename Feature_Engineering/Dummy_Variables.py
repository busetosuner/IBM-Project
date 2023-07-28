import pandas as pd
import numpy as np

# Dummy variables 

# if there are equal or less than 3 unique values turn into dummy variables
def create_dummies(df, header):
    # Create new dataframe for dummy variables
    dummy_variable = pd.get_dummies(df[header], prefix=header, drop_first=True) # To avoid dummy variable trap drop_first is true

    # Turn True/False into 1/0 
    dummy_variable = dummy_variable.astype(int)

    # Add to main dataframe
    df = pd.concat([df, dummy_variable], axis=1)

    # Drop the original column
    df.drop(header, axis=1, inplace=True)

    return df
