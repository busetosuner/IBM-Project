import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def is_numeric(col):
    try:
        pd.to_numeric(col)
        return True
    except:
        return False
    

# Import Data Frame

file_path = r"C:\Users\hacer\OneDrive\Masaüstü\IBM\datasets\df.csv"

df = pd.read_csv(file_path, index_col= False)

# Get headers
headers = df.columns.values


# Give information about data frame 
get_information = input("Do you want to get a brief information about data? (Y/N) ")

if get_information.upper() == 'Y':
    print(df.describe(include='all'))

# Give information(number of elements, name, type) about specific column.
get_information = input("Do you want to get a brief information about an attribute of data? (Y/N) ")

if get_information.upper() == 'Y':
    for i, header in enumerate(headers):
        print(i," ", header)

    feature =  headers[int(input("Please enter the index of attribute you want: "))]    
    # print(df[feature].value_counts())

    target = headers[int(input("Please enter index of target attribute: "))]

    if (not (is_numeric(df[feature]))) and (not (is_numeric(df[target]))):
        print("Sorry, we cannot perform this request!")
    else:
        sns.boxplot(x = feature, y = target, data= df)
        plt.show()

# string top 10, numeric binning


""" 


"""



"""     if is_numeric(df[feature]):    
        target = headers[int(input("Please enter index of target attribute: "))]
        if is_numeric(df[target]):
            

    feature_counts = df[feature].value_counts().to_frame()
    feature_counts.rename(columns = {"": feature, "count" : 'value_counts'}, inplace=True)
    # feature_counts.index.name = feature
    print(feature_counts.head()) """

# Show dataframe by attributes.
get_information = input("Would you like to see the data by attributes you want? (Y/N) ")

if get_information.upper() == 'Y':
    for i, header in enumerate(headers):
        print(i," ", header)
    group_list = []
    group_list = [headers[int(item)] for item in input("Please enter the index of attributes you want: ").split()]
    
    target = headers[int(input("Please enter index of target attribute: "))]
    
    while not is_numeric(df[target]):
        print("Please enter numerical target: ")
        target = headers[int(input("Target: "))]
    
    group_list.append(target)
    df_test = df[group_list]
    group_list.pop()
    group_df = df_test.groupby(group_list, as_index=False).mean()

    index = headers[int(input("Please enter index of attribute you want to categorize: "))]
    columns = headers[int(input("Please enter another index of attribute you want to categorize: "))]
    grouped_pivot = group_df.pivot(index=index, columns=columns)
    print(grouped_pivot)

# The visualization features will be added...
