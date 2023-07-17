
# DEA  ----- Descriptive Statistics/// Group By /// ANOVA /// Correlation


# ANOVA

# Her bir kategorik satırdaki columnları karşılaştırdı ama kullanıcıya sorarak da ilerleyebiliriz hangi sütunları karşılaştırmak istiyorsun diye

# Column bulma categorical için
categorical_columns = df_new.select_dtypes(include='object').columns

# Column bulma numerical data icin
numeric_columns = df_new.select_dtypes(include='number').columns

for column in categorical_columns:
    groups = df_new[column].unique() # Columndaki grupları alma
    
    # Gruplar arasında istatistiksel olarak anlamlı farklılık testi
    
    group_data = [df_new[df_new[column] == group][numeric_columns] for group in groups] # Sayısal değişkeni ve grupları kullanarak verileri oluşturma
    
    stat, p_value = f_oneway(*group_data) # Anova analizi
    
    print(f"Column: {column} | ANOVA Stats: {stat} | p-value: {p_value}")


# Görselleştirmek için çok seçenek var: Çubuk grafiği, kutu grafikleri vs. Çubuk kullandım



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
