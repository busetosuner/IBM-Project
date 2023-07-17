 
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
