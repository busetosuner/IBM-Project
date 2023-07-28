import pandas as pd
from sklearn.preprocessing import StandardScaler, scale
from factor_analyzer import FactorAnalyzer
import Numerical_Data
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import Numerical_Data

import statsmodels.api as sm
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression

def prepare(df, target):
    df_numeric = df[Numerical_Data.numeric_columns(df)]

    if target in df_numeric.columns:
        df_numeric = df_numeric.drop(target, axis = 1)
    return df_numeric

# to slow

def sfs(df, target, k):
    y = df[target]
    X = prepare(df, target)
    sfs = SFS( LinearRegression(), k_features = k, forward=True, floating=False, scoring='r2', cv = 0)
    sfs.fit(X,y)
    print(sfs.k_feature_names_)

def forward_selection(df, target, significance_level = 0.05):
    y = df[target]
    df = prepare(df, target)
    initial_features = df.columns.tolist()
    best_features = []

    while(len(initial_features)>0):
        remaining_features = list(set(initial_features)- set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column  in remaining_features:
            model = sm.OLS(y, sm.add_constant(df[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_pvalue = new_pval.min()
        if(min_pvalue < significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
        return best_features

def backward_elimination(df, target, significance_level = 0.05):
    y = df[target]
    df = prepare(df, target)
    features = df.columns.tolist()

    while(len(features)>0):
        features_with_constant = sm.add_constant(df[features])
        p_values = sm.OLS(y, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break        
    return features
    
def eliminate_with_corr(df):
    corr_matrix = df.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    print("\n To drop by corr_matrix: ",to_drop)
    return to_drop

def factor_analysis(df, threshold = 0.5):
    no_retains = len(df.axes[0])
    fa = FactorAnalyzer(n_factors=no_retains, rotation='varimax')
    fa.fit(df)

    factor_loadings = fa.loadings_
    explained_variance = fa.get_factor_variance()[1]
    # print("Explained_variance of FA:\n ", explained_variance)
    cumulative_variance = np.cumsum(explained_variance)

    sum = 0
    for i in range(len(factor_loadings)):
        sum += abs(factor_loadings[i][0])
    
    threshold = sum / len(factor_loadings)

    # print("\nLimit: ",threshold)
    
    selected_columns = df.columns[abs(factor_loadings[0]) >= threshold].tolist()

    new_factor = pd.Series(abs(factor_loadings[:,0]))
    
    if (len(selected_columns) == 0):
        selected_columns.append(df.columns.tolist()[new_factor.argmax()])

    return selected_columns, cumulative_variance

plt.style.use('default')
 
# 
def pc_analysis(df, threshold=0.5):
    if(len(df.axes[0]) < 2):
        return [], 0
    X = scale(df)
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)

    scores = pca.transform(X)
    scores_df = pd.DataFrame(scores, columns=["PC1", "PC2"])
    # print("Scores df:\n ", scores_df.head())

    loadings = pca.components_.T
    dfa_loadings = pd.DataFrame(loadings, columns=["PC1", "PC2"], index=df.columns.values)
    # print("Dfa loadings: \n",dfa_loadings.head())
    column_names = list(dfa_loadings.index)
    selected_columns = []
    for i, loading in enumerate(dfa_loadings["PC1"]):
        if abs(loading) >= threshold:
            selected_columns.append(column_names[i])

    if (len(selected_columns) == 0):
        selected_columns.append(dfa_loadings["PC1"].abs().idxmax())
    explained_variance = pca.explained_variance_ratio_
    # print("\nExplained variance: \n", explained_variance)

    explained_variance = np.insert(explained_variance, 0, 0)
    cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))

    pc_df = pd.DataFrame(["", "PC1", "PC2", "PC3"], columns =["PC"])
    explained_variance_df = pd.DataFrame(explained_variance, columns=["Explained_Variance"])
    cumulative_variance_df = pd.DataFrame(cumulative_variance, columns=["Cumulative_Variance"])

    df_explained_varience = pd.concat([pc_df, explained_variance_df, cumulative_variance_df], axis = 1)
    
    return selected_columns, cumulative_variance
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
    cumulative_variance = np.cumsum(explained_variance)

    n_components = np.argmax(explained_variance >= threshold) + 1

    reduced_data = np.dot(df, V_t[:n_components].T)
    # Create a new DataFrame with the reduced number of features
    column_names = [f'component_{i+1}' for i in range(n_components)]
    df_reduced = pd.DataFrame(reduced_data, columns=column_names)

    return df_reduced, cumulative_variance


def decide(df, target):
    to_drop = eliminate_with_corr(df)
    df.drop(to_drop, axis=1, inplace= True)
    """ 
    selectedbyForward = forward_selection(df, target)
    selectedbyBackward = backward_elimination(df, target)
    
    print("Forward Selection: ", selectedbyForward)

    print("Backward Elimination: ", selectedbyBackward) """
    target_col = df[target]
    df = prepare(df, target)
    selectedbyPCA, pc_cumulative = pc_analysis(df)
    selectedbyFA, fa_cumulative = factor_analysis(df)
    # selectedbSVD, svd_cumulative = svd(df)

    # sfs = feature_selection.sfs(df, target, 5)

    print("Selected by PCA:", selectedbyPCA)
    print("Selected by FA:", selectedbyFA)

    print("Cumulative of PCA: ", pc_cumulative[-1])
    print("Cumulative of FA: ", fa_cumulative[-1])

    if(fa_cumulative[-1] > pc_cumulative[-1]):
        print("Featues are selected by FA")
        df = pd.concat([df[selectedbyFA], target_col], axis = 1)
    else:
        print("Featues are selected by PCA")
        df = pd.concat([df[selectedbyPCA], target_col], axis = 1)
    # print("Cumulative of SVD: ", svd_cumulative[-1])
    return df
