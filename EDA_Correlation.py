       
# CORRELATION

#Korelasyonun değeri -1 ile 1 arasında değişir. 1, mükemmel pozitif korelasyonu ifade ederken, -1 mükemmel negatif korelasyon
# 0 ise iki değişken arasında bir ilişki olmadığını gösterir.

# Burada kullanıcıya sorabiliriz korelasyon etmek istediğiniz sütunu seçin şeklinde

numeric_columns = df_new.select_dtypes(include='number').columns

# Korelasyon matrisini hesaplama
correlation_matrix = df_new[numeric_columns].corr()

#  Korelasyon matrisi, tüm sayısal sütunlar arasındaki ikili korelasyonları gösterir. Her sütunun diğer sütunlarla olan korelasyonunu görmek için matrisin tamamı

print(correlation_matrix)


#Görselleştirmek için


#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#plt.title('Korelasyon Matrisi')
#plt.show()
