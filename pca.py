from sklearn import decomposition
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, scale
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

plt.style.use('default')
 

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
