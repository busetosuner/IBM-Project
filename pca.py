from sklearn.preprocessing import scale
from sklearn import decomposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pca_analysis(df):
    X = scale(df)
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)

    scores = pca.transform(X)
    scores_df = pd.DataFrame(scores, columns=["PC1", "PC2", "PC3"])
    print("Scores df:\n ", scores_df.head())

    loadings = pca.components_.T
    dfa_loadings = pd.DataFrame(loadings, columns=["PC1", "PC2", "PC3"], index=df.columns.values)
    print("Dfa loadings: \n",dfa_loadings.head())

    explained_variance = pca.explained_variance_ratio_
    print("Explained variance: \n", explained_variance)

    explained_variance = np.insert(explained_variance, 0, 0)
    cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))

    pc_df = pd.DataFrame(["", "PC1", "PC2", "PC3"], columns =["PC"])
    explained_variance_df = pd.DataFrame(explained_variance, columns=["Explained_Variance"])
    cumulative_variance_df = pd.DataFrame(cumulative_variance, columns=["Cumulative_Variance"])

    df_explained_varience = pd.concat([pc_df, explained_variance_df, cumulative_variance_df], axis = 1)
    print("Pc DF: \n", df_explained_varience.head())
    
    
    
    """ 
    fig = px.bar(df_explained_varience, x = 'PC', y = "Explained_Variance", text= "Explained_Variance",width = 800)
    fig.update_traces(texttemplate="%{text:.3f}",textposition="outside")
    fig.show()

    fig3d = px.scatter_3d(scores_df, x = "PC1", y = "PC2", z = "PC3")
    fig3d.show() """