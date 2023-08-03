import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import Feature_Engineering.Binning as Binning

def numeric_columns(df):
    numeric_columns = []
    for column in df.columns:
         if column != 'Unnamed: 0' and Binning.is_numeric(df[column]):
            numeric_columns.append(column)

    return numeric_columns

# Outlier detection
def detect_outliers(data, threshold=3):
    outliers = []
    for column in data.columns:
        # Calculate Z-score 
        z_scores = (data[column] - data[column].mean()) / data[column].std()
        # Mark as outliers those greater than the threshold of the z-score
        outliers.extend(data[column][np.abs(z_scores) > threshold])
    return outliers

def drop_outliers(df):
    numeric_data = df[numeric_columns(df)]
    outliers = detect_outliers(numeric_data)

    # Calculate the percentage of outliers in the data set (to see the effect of outliers on the data set)
    outlier_percentage = len(outliers) / len(df) * 100

    print("Outlier persentage: %", outlier_percentage)
    #print("Dataset Columns: ", df_cleaned.columns)
 
    # Remove outliers
    df_cleaned_no_outliers = df.drop(pd.DataFrame(outliers).reset_index(drop=True))

    # print("Without outliers Data:")
    # print(df_cleaned_no_outliers.head())

    return df_cleaned_no_outliers

# NORMALIZATION
def normalization(df):
    numeric_Columns = numeric_columns(df)
    scaler = MinMaxScaler(feature_range=(0, 1))
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
