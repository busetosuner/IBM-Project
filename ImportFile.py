import os
import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
 # pip install scikit-learn yapmalısın


#Hangi data formatında kontrol ediyor
def check_data_format(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    # Creating empty dataFrame
    df = pd.DataFrame()

    data_format = ""
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
        data_format = 'csv'
    elif file_extension == '.xlsx' or file_extension == '.xls':
        df = pd.read_excel(file_path)
        data_format = 'excel'
    elif file_extension == '.json':
        df = pd.read_json(file_path)
        data_format = 'json'
    else:
        print("Error")
        sys.exit()

    print("Dataset:", data_format)
    return df

file_path = 'DataScienceSalaries.csv'  

df = pd.read_csv(file_path)
data_format = check_data_format(file_path)


# Eğer 3 data formatından biri değilse bitirecek
if data_format is None:
    print("Error")
    sys.exit()
else: # Burayı yazdırıyor burayla daha bir şey yaparız diye düşündüm ihtiyaç olursa
    print("Dataset:", data_format)
