import pandas as pd
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns

def perform_anova(df):
    # Column bulma categorical için
    categorical_columns = df.select_dtypes(include='object').columns

    # Column bulma numerical data icin
    numeric_columns = df.select_dtypes(include='number').columns

    results = []
    df_results = pd.DataFrame()  # Boş bir DataFrame oluşturuyoruz

    for column in categorical_columns:
        groups = df[column].unique()  # Columndaki grupları alma

        group_data = [df[df[column] == group][numeric_columns] for group in groups]  # Sayısal değişkeni ve grupları kullanarak verileri oluşturma

        stat, p_value = f_oneway(*group_data)  # Anova analizi

        result = {
            'Column': column,
            'ANOVA Stats': stat,
            'p-value': p_value
        }

        results.append(result)

    # Sonuçları DataFrame'e ekliyoruz
    df_results = pd.concat([df_results, pd.DataFrame(results)], ignore_index=True)

    # Filter out non-numeric rows from 'ANOVA Stats' column
    numeric_mask = df_results['ANOVA Stats'].apply(lambda x: pd.api.types.is_numeric_dtype(x))
    df_results = df_results[numeric_mask]

    # Remove problematic rows in 'ANOVA Stats' column
    df_results = df_results.dropna(subset=['ANOVA Stats'])

    try:
        # Convert the 'ANOVA Stats' column to numeric data type
        df_results['ANOVA Stats'] = pd.to_numeric(df_results['ANOVA Stats'])
    except Exception as e:
        print("Error converting 'ANOVA Stats' column to numeric data type:")
        print(e)

    # Convert the 'Column' column to categorical data type
    df_results['Column'] = pd.Categorical(df_results['Column'])

    # Görselleştirme için
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Column', y='ANOVA Stats', data=df_results)
    plt.xlabel('Column')
    plt.ylabel('ANOVA Stats')
    plt.title('ANOVA Results')
    plt.xticks(rotation=45)
    plt.show()

    return results
