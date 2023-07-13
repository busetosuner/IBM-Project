import os
import pandas as pd
import sys

def check_data_format(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension == '.csv':
        return 'csv'
    elif file_extension == '.xlsx' or file_extension == '.xls':
        return 'excel'
    elif file_extension == '.json':
        return 'json'
    else:
        return None

file_path = 'DataScienceSalaries.csv'  

data_format = check_data_format(file_path)

if data_format is None:
    print("Error")
    sys.exit()
else:
    print("Dataset:", data_format)
    
    
    
    
    
    
    
    
    
    
    
    
    
