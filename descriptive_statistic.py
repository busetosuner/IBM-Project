import pandas as pd
import numpy as np

# Import Data Frame

file_path = r"C:\Users\hacer\OneDrive\Masaüstü\IBM\datasets\ds_salaries.csv"

df = pd.read_csv(file_path, index_col= False)

# Get headers
headers = df.columns.values


# Give information about data frame 
get_information = input("Do you want to get a brief information about data? (Y/N) ")

if get_information.upper() == 'Y':
    print(df.describe())

# Give information(number of elements, name, type) about specific column.
get_information = input("Do you want to get a brief information about an attribute of data? (Y/N) ")

if get_information.upper() == 'Y':
    for i, header in enumerate(headers):
        print(i," ", header)
    index =  int(input("Please enter the index of attribute you want: "))
    feature = headers[index]
    print(df[feature].value_counts())


# Show dataframe by attributes.
get_information = input("Would you like to see the data by attributes you want? (Y/N) ")

if get_information.upper() == 'Y':
    for i, header in enumerate(headers):
        print(i," ", header)
    group_list = []
    group_list = [headers[int(item)] for item in input("Please enter the index of attributes you want: ").split()]
        
    group_df = df.groupby(group_list)
    print(group_df.first())

# The visualization features will be added...