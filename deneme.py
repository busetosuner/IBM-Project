import os
import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
 # pip install scikit-learn yapmalısın


#Hangi data formatında kontrol ediyor
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
  
# Numerical data ile başlıyorum

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


















