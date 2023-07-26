import pandas as pd
from sklearn.preprocessing import StandardScaler, scale
from factor_analyzer import FactorAnalyzer
import Numerical_Data
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# to slow
def eliminate_with_corr(df):
    corr_matrix = df.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

    print("To drop: ",to_drop)
    return to_drop

def factor_analysis(df, no_retains = 10, threshold = 0.5):

    fa = FactorAnalyzer(n_factors=no_retains, rotation='varimax')
    fa.fit(df)

    factor_loadings = fa.loadings_
    explained_variance = fa.get_factor_variance()[1]
    print("Explained_variance of FA:\n ", explained_variance)

    sum = 0
    for i in range(len(factor_loadings)):
        sum += abs(factor_loadings[i][0])
    
    threshold = sum / len(factor_loadings)

    print("\nLimit: ",threshold)
    
    selected_columns = df.columns[abs(factor_loadings)[0] >= threshold]
    df_selected = df[selected_columns]
    print("Selected FA: ", df_selected.columns.values)
    return df_selected

plt.style.use('default')
 
# 
def pc_analysis(df, thereshold=0.5):
    X = scale(df)
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)

    scores = pca.transform(X)
    scores_df = pd.DataFrame(scores, columns=["PC1", "PC2", "PC3"])
    print("Scores df:\n ", scores_df.head())

    loadings = pca.components_.T
    dfa_loadings = pd.DataFrame(loadings, columns=["PC1", "PC2", "PC3"], index=df.columns.values)
    print("Dfa loadings: \n",dfa_loadings.head())
    column_names = list(dfa_loadings.index)
    selected_columns = []
    for i, loading in enumerate(dfa_loadings["PC1"]):
        if abs(loading) >= thereshold:
            selected_columns.append(column_names[i])
    
    print(selected_columns)

    explained_variance = pca.explained_variance_ratio_
    # print("\nExplained variance: \n", explained_variance)

    explained_variance = np.insert(explained_variance, 0, 0)
    cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))

    pc_df = pd.DataFrame(["", "PC1", "PC2", "PC3"], columns =["PC"])
    explained_variance_df = pd.DataFrame(explained_variance, columns=["Explained_Variance"])
    cumulative_variance_df = pd.DataFrame(cumulative_variance, columns=["Cumulative_Variance"])

    df_explained_varience = pd.concat([pc_df, explained_variance_df, cumulative_variance_df], axis = 1)
    print("Pc DF: \n", df_explained_varience.head())
    
    """ 
    Visualization
    plt.bar(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_)

    plt.plot(range(1, len(pca.explained_variance_) + 1), np.cumsum(pca.explained_variance_), c="red", label = "Cumulative Explained Varience")
    
    plt.legend(loc = "upper left")
    plt.xlabel('PCA Feature')
    plt.ylabel('Explained variance')
    plt.title('Feature Explained Variance')

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    y_data = X[:,1]
    z_data = X[:,2]
    x_data = X[:,0]
    
    ax.scatter3D(x_data, y_data, z_data, c=z_data)

    # Plot title of graph
    plt.title(f'3D Scatter of Data')
    
    # Plot x, y, z even ticks
    ticks = np.linspace(-3, 3, num=5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    plt.show() """

# Singular Value Decomposition
def svd(df, threshold = 0.5 ):
    df = Numerical_Data.standardization(df)

    U, S, V_t = np.linalg.svd(df, full_matrices=False)

    explained_variance = np.cumsum(S ** 2) / np.sum(S ** 2)
    print("Explained variance of svd \n", explained_variance)

    n_components = np.argmax(explained_variance >= threshold) + 1

    reduced_data = np.dot(df, V_t[:n_components].T)
    # Create a new DataFrame with the reduced number of features
    column_names = [f'component_{i+1}' for i in range(n_components)]
    df_reduced = pd.DataFrame(reduced_data, columns=column_names)

    return df_reduced
