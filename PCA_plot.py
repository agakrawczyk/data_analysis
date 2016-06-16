"""
creates numpy array with the  random data - the features of the created data can be changed
centers and scales the data from the array and then calculates the PCA
clusters the PCA output using 3 different methods: KMeans, Affinity Propagation and MeanShift
Plots 4 graphs containing PCA and 3 graphs representing different clustering methods.
Prints silhouette scores for every clustering method in the title of the graph.

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from math import sqrt
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, estimate_bandwidth
import sklearn.metrics


def scale_data(matrix_in):
    """

    :param matrix_in: numpy array
    :return: matrix with scaled and centered data
    """

    dataMatrix = np.transpose(matrix_in)

    rowmeans = dataMatrix.mean(axis=1)

    centered_data = np.array([row - b for row, b in zip(dataMatrix, rowmeans)])

    rowstds = centered_data.std(axis=1)

    scaled_data = np.array([row/sqrt(b) if b != 0 else row for row, b in zip(centered_data, rowstds)])

    final_matrix = np.transpose(scaled_data)

    return final_matrix


def best_cluster_nr(pca_data):
    """

    :param pca_data: data after PCA analysis
    calculates the silhouette scores for the different numbers of clusters (from 2 to 10) and compares the results
    1 is the best one, -1 the worst
    :return: the best cluster number based on the calculated silhouette score
    """

    best_silhouette = -1
    best_cluster_number = 2

    for i in range(2, 10):
        k_means = KMeans(n_clusters=i)
        fit = k_means.fit(pca_data)
        silhouette_score = sklearn.metrics.silhouette_score(pca_data, fit.labels_, metric='euclidean')

        if silhouette_score > best_silhouette:
            best_silhouette = silhouette_score
            best_cluster_number = i

    print('Optimal number of KMeans clusters', best_cluster_number)

    return best_cluster_number


if __name__ == '__main__':

    random = np.random.normal(10, size=(100, 5))

    groups = np.array(['a']*25 + ['b']*25 + ['c']*25 + ['d']*25)

    random[groups == 'a', 2] += 5
    random[groups == 'b', 2] -= 5
    random[groups == 'c', 1] += 5

    final_matrix = scale_data(random)

    sklearn_pca = sklearnPCA(n_components=2)
    sklearn_transf = sklearn_pca.fit_transform(final_matrix)

    best_cluster_number = best_cluster_nr(sklearn_transf)

    # Clusters the data after sklearn PCA using 3 different methods

    k_means = KMeans(n_clusters=best_cluster_number)
    fit = k_means.fit(sklearn_transf)

    af = AffinityPropagation(damping=0.7)
    affprop = af.fit(sklearn_transf)

    bndwth = estimate_bandwidth(sklearn_transf, quantile=0.2, n_samples=100)
    ms = MeanShift(bandwidth=bndwth, bin_seeding=True)
    meanshift = ms.fit(sklearn_transf)

    # Calculates the silhouette score for every clustering method

    kmeans_silh = sklearn.metrics.silhouette_score(sklearn_transf, fit.labels_, metric='euclidean')
    affinity_propagation_silh = sklearn.metrics.silhouette_score(sklearn_transf, affprop.labels_, metric='euclidean')
    mean_shift_silh = sklearn.metrics.silhouette_score(sklearn_transf, meanshift.labels_, metric='euclidean')

    colors = {'a': 'blue', 'b': 'green', 'c': 'yellow', 'd': 'red'}
    cluster_colors = {0: 'purple', 1: 'pink', 2: 'grey', 3: 'cyan', 4: 'black', 5: 'brown', 6: 'orange', 7: 'darkblue'}

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    for group, pca, cluster, aff, mnshf in zip(groups, sklearn_transf, fit.labels_, affprop.labels_, meanshift.labels_):

        ax1.plot(pca[0], pca[1], 'o', markersize=7,
                 c=colors[group] if group in colors.keys() else 'gray',
                 label=group,
                 alpha=0.5)
        ax1.set_title('PCA colored by groups', size=7)
        ax2.plot(pca[0], pca[1], 'o', markersize=7,
                 c=cluster_colors[cluster] if cluster in cluster_colors.keys() else 'grey',
                 label=cluster,
                 alpha=0.5)
        ax2.set_title('PCA KMeans colored by clusters, silhouette number=' + str(kmeans_silh), size=7)
        ax3.plot(pca[0], pca[1], 'o', markersize=7,
                 c=cluster_colors[aff] if aff in cluster_colors.keys() else 'grey',
                 label=aff,
                 alpha=0.5)
        ax3.set_title('PCA Affinity Propagation colored by clusters, silhouette number=' +
                      str(affinity_propagation_silh), size=7)
        ax4.plot(pca[0], pca[1], 'o', markersize=7,
                 c=cluster_colors[mnshf] if mnshf in cluster_colors.keys() else 'grey',
                 label=mnshf,
                 alpha=0.5)
        ax4.set_title('PCA Mean Shift colored by clusters, silhouette number=' + str(mean_shift_silh), size=7)

    ax3.set_xlabel('Principal Component 1')
    ax3.set_ylabel('Principal Component 2')
    plt.draw()
    plt.show()


