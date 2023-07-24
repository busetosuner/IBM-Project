import pandas as pd
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
import Numerical_Data

def feature_selection(df, target, no_retains = 10, limit = 0.5):

    fa = FactorAnalyzer(n_factors=no_retains, rotation='varimax')
    fa.fit(df)

    print(fa)

    factor_loadings = fa.loadings_

    print(factor_loadings)

    sum = 0

    for i in range(len(factor_loadings)):
        sum += abs(factor_loadings[i][0])
    
    limit = sum / len(factor_loadings)

    print("\nLimit: ",limit)
    
    selected_columns = df.columns[abs(factor_loadings)[0] >= limit]
    df_selected = df[selected_columns]

    return df_selected
