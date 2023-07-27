import matplotlib.pyplot as plt
import seaborn as sns

def calculate_correlation(df, target):
    # Korelasyon matrisi için sayısal sütunları seç
    numeric_columns = df.select_dtypes(include='number').columns

    # Korelasyon matrisini hesaplama
    correlation_matrix = df[numeric_columns].corr()

    # Sadece hedef değişken ile olan korelasyonu al
    target_correlation = correlation_matrix[target]
    
    #print("\nTarget ({}):".format(target))
    #print(target_correlation)

    plt.figure(figsize=(13, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Korelasyon Matrisi')
    plt.show()  

    highly_corr = df.columns[abs(target_correlation) >= 0.5].tolist()

    print("This attiributes are highly correlated with target: \n",highly_corr)
    return target_correlation
    
    #Görselleştirmek için

                    
