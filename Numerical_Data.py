import pandas as pd
import numpy as np
import Binning
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def numeric_columns(df):
    numeric_columns = []
    for column in df.columns:
         if column != 'Unnamed: 0' and Binning.is_numeric(df[column]):
            numeric_columns.append(column)

    return numeric_columns

# Outlier tespiti
def detect_outliers(data, threshold=3):
    outliers = []
    for column in data.columns:
        # Z-puanı için
        z_scores = (data[column] - data[column].mean()) / data[column].std()
        # Z-puanının threshold değerinden büyük olanları aykırı değer olarak işaretle
        outliers.extend(data[column][np.abs(z_scores) > threshold])
    return outliers

def drop_outliers(df):
    numeric_data = df[numeric_columns(df)]
    outliers = detect_outliers(numeric_data)

    # Outlierların veri setindeki yüzdesini hesaplamak (outlierların veri setine etkisini görmek için)
    outlier_percentage = len(outliers) / len(df) * 100

    print("Outlier persentage: %", outlier_percentage)
    #print("Veri Seti Sütunlari:", df_cleaned.columns)
 
    # Outlierları çıkarmak için
    df_cleaned_no_outliers = df.drop(pd.DataFrame(outliers).reset_index(drop=True))

    # print("Without outliers Data:")
    # print(df_cleaned_no_outliers.head())

    return df_cleaned_no_outliers

# NORMALIZATION
def normalization(df):
    numeric_Columns = numeric_columns(df)
    scaler = MinMaxScaler(feature_range=(0, 10))
    for column in numeric_Columns:
        df[column] = scaler.fit_transform(df[[column]])

    # print("Normalized Data:")
    # print(df.head())

    return df


# STANDARDIZATION
def standardization(df):
    numeric_Columns = numeric_columns(df)
    scaler = StandardScaler()
    for column in numeric_Columns:
        df[column] = scaler.fit_transform(df[[column]])
    # print("Standardized Data:")
    # print(df.head())

    return df   
