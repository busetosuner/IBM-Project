
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
    
