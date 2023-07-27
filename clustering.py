import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from IPython.display import clear_output

def plot_clusters(df, labels, centroids=None, iteration=None):
    if len(df.columns) <= 1:
        print("Data has only one feature. Cannot perform PCA for 2 components.")
        return

    pca = PCA(n_components=2)
    df_2d = pca.fit_transform(df)
    if centroids is not None:
        # Reshape centroids to 2D array if it's not already
        centroids = centroids.values.reshape(1, -1)
        centroids_2d = pca.transform(centroids)
    clear_output(wait=True)
    plt.title(f'Iteration {iteration}' if iteration else 'Final Result')
    plt.scatter(x=df_2d[:, 0], y=df_2d[:, 1], c=labels)
    if centroids is not None:
        plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1], marker='X', s=200, c='black')
    plt.pause(0.5 if iteration else 3)


def get_labels(df, model):
    if isinstance(model, KMeans):
        return model.labels_
    elif isinstance(model, AgglomerativeClustering):
        return model.fit_predict(df)
    else:
        raise ValueError("Unknown model type!")


def cluster(df, k):
    if len(df.columns) <= 1:
        print("Data has only one feature. Cannot perform clustering.")
        return

    # K-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10)  # Set the value of n_init explicitly
    kmeans.fit(df)
    kmeans_labels = get_labels(df, kmeans)

    # Hierarchical Clustering (Agglomerative)
    hierarchical = AgglomerativeClustering(n_clusters=k)
    hierarchical_labels = get_labels(df, hierarchical)

    # Plot initial data
    plot_clusters(df, kmeans_labels)

    # K-means clustering with animation
    old_centroids = None
    max_iteration = 100
    

    for iteration in range(1, max_iteration + 1):
        centroids = df.apply(lambda x: x.sample().iloc[0])
        kmeans_labels = get_labels(df, kmeans)
    
        if centroids.equals(old_centroids):
            break
        old_centroids = centroids
        plot_clusters(df, kmeans_labels, centroids, iteration)
    plot_clusters(df, kmeans_labels, centroids)

    # Hierarchical Clustering (Agglomerative) with the final result
    plot_clusters(df, hierarchical_labels)

    # Compare the two algorithms and select the better one
    kmeans_sse = kmeans.inertia_
    hierarchical_sse = np.sum((df - np.array(old_centroids.T)) ** 2)  # Using the last K-means centroids as hierarchical centroids
    print("K-Means SSE:", kmeans_sse)
    print("Hierarchical Clustering SSE:", hierarchical_sse)

    if kmeans_sse < hierarchical_sse.values[0]:  # Accessing the scalar value from the Series
        print("The better algorithm is: K-Means")
    else:
        print("The better algorithm is: Hierarchical Clustering (Agglomerative)")
