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


# DUPLICATE BOLUMU BASLANGICI
# Eğer duplicate varsa sayması için
def count_duplicates(df):
    # Satırda yinelenen varsa
    duplicated_rows = df.duplicated()
    num_duplicates_rows = duplicated_rows.sum()

    # Sütunda yinelenen varsa
    duplicated_columns = df.T.duplicated()
    num_duplicates_columns = duplicated_columns.sum()

    # Sütunlar arasında yinelenen satırlar varsa
    duplicated_rows_among_columns = df.duplicated(keep=False)
    num_duplicates_rows_among_columns = duplicated_rows_among_columns.sum()

    # Tamamen aynı olan satırlar varsa
    duplicated_rows_all_columns = df.duplicated()
    num_duplicates_rows_all_columns = duplicated_rows_all_columns.sum()

    return num_duplicates_rows, num_duplicates_columns, num_duplicates_rows_among_columns, num_duplicates_rows_all_columns

num_duplicates_rows, num_duplicates_columns, num_duplicates_rows_among_columns, num_duplicates_rows_all_columns = count_duplicates(df)

# Toplam duplicate sayısı hesaplama
total_duplicates = num_duplicates_rows + num_duplicates_columns + num_duplicates_rows_among_columns + num_duplicates_rows_all_columns

print("Duplicate number:", total_duplicates)




# Burası da temizleme kısmı
def clean_duplicates(df):
    # Satır düzeyinde yinelenen varsa
    duplicated_rows = df.duplicated()

    # Sütun düzeyinde yinelenen varsa
    duplicated_columns = df.T.duplicated()

    # Sütunlar arasında yinelenen varsa
    duplicated_rows_among_columns = df.duplicated(keep=False)

    # Tamamen aynı olan satırlar varsa
    duplicated_rows_all_columns = df.duplicated()

    # Yinelenen değerleri düzeltme fonksiyonu
    df_cleaned = df.drop_duplicates()

    return df_cleaned


df_cleaned= clean_duplicates(df)


# Temizlenmiş DataFrame'i kontrol etme
print(df_cleaned.shape)  # DataFrame'in boyutunu yazdırma

# Temizlendikten sonra sorun var mı check'i
total_duplicates_after_cleaning = count_duplicates(df_cleaned)

print("Duplicate number after cleaning:", sum(total_duplicates_after_cleaning))
    
    

# Categorical ve Numerical Data'yı ayırma bölümü
  
# NUMERICAL DATA

# Öncelikle outlierlari belirliyorum


numeric_columns = df_cleaned.select_dtypes(include=np.number).columns

# Numerik veri kümesini almak
numeric_data = df_cleaned[numeric_columns]

# Outlier tespiti
def detect_outliers(data, threshold=3):
    outliers = []
    for column in data.columns:
        # Z-puanı için
        z_scores = (data[column] - data[column].mean()) / data[column].std()
        # Z-puanının threshold değerinden büyük olanları aykırı değer olarak işaretle
        outliers.extend(data[column][np.abs(z_scores) > threshold])
    return outliers

outliers = detect_outliers(numeric_data)

# Outlierları çıkarmak için
df_cleaned_no_outliers = df_cleaned.drop(pd.DataFrame(outliers).reset_index(drop=True))

# Outlierların veri setindeki yüzdesini hesaplamak (outlierların veri setine etkisini görmek için)
outlier_percentage = len(outliers) / len(df_cleaned) * 100


print("Outliers:", outliers)
#print("Veri Seti Sütunlari:", df_cleaned.columns)

# Normalization
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df_cleaned_no_outliers[numeric_columns])
df_normalized = pd.DataFrame(normalized_data, columns=numeric_columns)

# Standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df_cleaned_no_outliers[numeric_columns])
df_standardized = pd.DataFrame(standardized_data, columns=numeric_columns)

print("Normalized Data:")
print(df_normalized.head())

print("Standardized Data:")
print(df_standardized.head())


# CATEGORICAL DATA
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

path = r'C:\Users\User\OneDrive\Masaüstü\IBM-datasets' #Buse
# path = 'C:\\Users\\hacer\\OneDrive\\Masaüstü\\IBM\\datasets\\' #Hacer


df_new.to_csv(path + "df.csv", index=False)







#HANDLING MISSING DATA

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


print(df_new)










