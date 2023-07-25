import pandas as pd

def calculate_correlation(df, target):
    # Korelasyon matrisi için sayısal sütunları seç
    numeric_columns = df.select_dtypes(include='number').columns

    # Korelasyon matrisini hesaplama
    correlation_matrix = df[numeric_columns].corr()

    # Sadece hedef değişken ile olan korelasyonu al
    target_correlation = correlation_matrix[target]

    return target_correlation



#Görselleştirmek için


#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#plt.title('Korelasyon Matrisi')
#plt.show()                   
