import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from IPython.display import clear_output
import handle_missing_values

def plot_clusters(df, labels, centroids, iteration):
    pca = PCA(n_components=2)
    df_2d = pca.fit_transform(df)
    centroids_2d = pca.fit_transform(centroids.T)
    clear_output(wait = True)
    plt.title(f'Iteration{iteration}')
    plt.scatter(x=df_2d[:,0], y= df_2d[:,1], c = labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
    plt.pause(0.05)

def plot_clusters_scatter(df, labels, centroids, iteration):
    centroid_T= centroids.T
    plt.title(f'Iteration{iteration}')
    plt.scatter(x=df.iloc[:,0], y= df.iloc[:,1], c = labels)
    plt.scatter(x=centroid_T.iloc[:,0], y=centroid_T.iloc[:,1])
    plt.pause(0.05)

def random_centroids(df, k):
    centroids = []
    for i in range(k):
        centroid = df.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)

def get_labels(df, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((df - x)**2).sum(axis=1)))
    return distances.idxmin(axis=1)

def new_centroids(df, labels, k):
    np.seterr(divide = 'ignore')
    centroids = df.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return centroids

def cluster(df, k):
    X = df 
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(X)

    identified_clusters = kmeans.fit_predict(X)
    # print("Identified Clusters: ", identified_clusters)

    data_with_clusters = df.copy()
    data_with_clusters['Clusters'] = identified_clusters

    centroids = random_centroids(df, k)
    # print("Centroids: \n", centroids)

    labels = get_labels(df, centroids)
    # print("Labels counts: ", labels.value_counts())

    old_centroids = pd.DataFrame()

    max_iteration = 100
    iteration = 1  

    while iteration < max_iteration and not centroids.equals(old_centroids):
        old_centroids = centroids
        labels = get_labels(df, centroids)
        centroids = new_centroids(df, labels, k)
        centroids = handle_missing_values.clean_missing(centroids)
        # print("Centroids: \n", centroids)
        plot_clusters(df, labels, centroids, iteration)
        iteration += 1
    plt.show()

"""   
    plt.scatter(df.iloc[:,0], df.iloc[:,1], c=data_with_clusters['Clusters'], cmap='rainbow')
    plt.show()    
 """
