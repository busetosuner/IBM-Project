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


def print_duplicate(df):
    num_duplicates_rows, num_duplicates_columns, num_duplicates_rows_among_columns, num_duplicates_rows_all_columns = count_duplicates(df)

    # Toplam duplicate sayısı hesaplama
    total_duplicates = num_duplicates_rows + num_duplicates_columns + num_duplicates_rows_among_columns + num_duplicates_rows_all_columns

    print("Duplicate number:", total_duplicates)


def check_duplicate(df):
    
    # Temizlenmiş DataFrame'i kontrol etme
    print(df.shape)  # DataFrame'in boyutunu yazdırma

    # Temizlendikten sonra sorun var mı check'i
    total_duplicates_after_cleaning = count_duplicates(df)

    print("Duplicate number after cleaning:", sum(total_duplicates_after_cleaning))
             


# Burası da temizleme kısmı
def clean_duplicates(df):

    print_duplicate(df)
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

    check_duplicate(df_cleaned)

    return df_cleaned
