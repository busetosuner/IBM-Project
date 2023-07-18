import os
import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
 # pip install scikit-learn yapmalısın
from scipy.stats import f_oneway
import seaborn as sns
import matplotlib.pyplot as plt





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


def print_duplicate():
    # Toplam duplicate sayısı hesaplama
    total_duplicates = num_duplicates_rows + num_duplicates_columns + num_duplicates_rows_among_columns + num_duplicates_rows_all_columns

    print("Duplicate number:", total_duplicates)

print_duplicate()


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

def check_duplicate():
    # Temizlenmiş DataFrame'i kontrol etme
    print(df_cleaned.shape)  # DataFrame'in boyutunu yazdırma

    # Temizlendikten sonra sorun var mı check'i
    total_duplicates_after_cleaning = count_duplicates(df_cleaned)

    print("Duplicate number after cleaning:", sum(total_duplicates_after_cleaning))
        
check_duplicate()       


# DUPLICATE SONU













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


def print_outliers():
    global outliers, df_cleaned_no_outliers, outlier_percentage
    
    
    outliers = detect_outliers(numeric_data)

    # Outlierları çıkarmak için
    df_cleaned_no_outliers = df_cleaned.drop(pd.DataFrame(outliers).reset_index(drop=True))

    # Outlierların veri setindeki yüzdesini hesaplamak (outlierların veri setine etkisini görmek için)
    outlier_percentage = len(outliers) / len(df_cleaned) * 100


    print("Outliers:", outliers)
    #print("Veri Seti Sütunlari:", df_cleaned.columns)


print_outliers()








# NORMALIZATION

def normalization():
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df_cleaned_no_outliers[numeric_columns])
    df_normalized = pd.DataFrame(normalized_data, columns=numeric_columns)
    print("Normalized Data:")
    print(df_normalized.head())

normalization()



# STANDARDIZATION

def standardization():
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df_cleaned_no_outliers[numeric_columns])
    df_standardized = pd.DataFrame(standardized_data, columns=numeric_columns)
    print("Standardized Data:")
    print(df_standardized.head())

standardization()




# CATEGORICAL DATA
# Get headers

def categorical_dummy():
    global headers, df_new
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


categorical_dummy()








#HANDLING MISSING DATA



def missing_data():
    global df_new
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
    
    
missing_data()    





# Missing strings and numeric values

def is_numeric(col):
    try:
        pd.to_numeric(col)
        return True
    except:
        return False

# because of creating copy of dataframe, prevent the chained assignment error 
def prevent():
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


prevent()






# DEA  ----- Descriptive Statistics/// Group By /// ANOVA /// Correlation


# ANOVA

# Her bir kategorik satırdaki columnları karşılaştırdı ama kullanıcıya sorarak da ilerleyebiliriz hangi sütunları karşılaştırmak istiyorsun diye

# Column bulma categorical için



def anova():
    global categorical_columns
    categorical_columns = df_new.select_dtypes(include='object').columns

    # Column bulma numerical data icin
    numeric_columns = df_new.select_dtypes(include='number').columns

    for column in categorical_columns:
        groups = df_new[column].unique() # Columndaki grupları alma
        
        # Gruplar arasında istatistiksel olarak anlamlı farklılık testi
        
        group_data = [df_new[df_new[column] == group][numeric_columns] for group in groups] # Sayısal değişkeni ve grupları kullanarak verileri oluşturma
        
        stat, p_value = f_oneway(*group_data) # Anova analizi
        
        print(f"Column: {column} | ANOVA Stats: {stat} | p-value: {p_value}")


anova()

# Görselleştirmek için çok seçenek var: Çubuk grafiği, kutu grafikleri vs.


def graphic_anova():
    for column in categorical_columns:
        groups = df_new[column].unique()
        anova_stats = []
        
        for group in groups:
            data = df_new[df_new[column] == group][numeric_columns]
            stat, p_value = f_oneway(*data.T.values)
            anova_stats.append(stat)
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=groups, y=anova_stats)
        plt.xlabel(column)
        plt.ylabel('ANOVA Stats')
        plt.title(f'ANOVA: {column}')
        plt.show()
        
        
        
        
        
# CORRELATION

#Korelasyonun değeri -1 ile 1 arasında değişir. 1, mükemmel pozitif korelasyonu ifade ederken, -1 mükemmel negatif korelasyon
# 0 ise iki değişken arasında bir ilişki olmadığını gösterir.

# Burada kullanıcıya sorabiliriz korelasyon etmek istediğiniz sütunu seçin şeklinde


def correlation():
    global correlation_matrix
    numeric_columns = df_new.select_dtypes(include='number').columns

    # Korelasyon matrisini hesaplama
    correlation_matrix = df_new[numeric_columns].corr()

    #  Korelasyon matrisi, tüm sayısal sütunlar arasındaki ikili korelasyonları gösterir. Her sütunun diğer sütunlarla olan korelasyonunu görmek için matrisin tamamı

    print(correlation_matrix)

    # Görselleştirmek icin
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Korelasyon Matrisi')
    plt.show()
    
correlation()
