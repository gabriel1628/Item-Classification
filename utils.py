import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, manifold, metrics


def ARI_fct(data, n_labels, cat_labels):
    """
    Function to determine t-SNE, determine clusters and compute ARI between clusters and real categories.

    Positional arguments :
    ----------------------

    encoded_data : nd_array
        text data encoded into a ndarray

    n_labels : int
        number of categories

    cat_labels : 1darray or list
        real category labels for each point

    return
    ------

    ARI : float
        ARI score

    X_tsne : 2darray
        embedding of the training data in 2-dimensional space

    cls.labels_ : ndarray
        cluster labels of each point
    """
    start = time.time()

    tsne = manifold.TSNE(
        n_components=2,
        perplexity=30,
        n_iter=2000,
        init="random",
        learning_rate=200,
        random_state=42,
    )
    X_tsne = tsne.fit_transform(data)

    # Infering clusters from t-SNE data
    cls = cluster.KMeans(n_clusters=n_labels, n_init=100, random_state=42)
    cls.fit(X_tsne)

    ARI = np.round(metrics.adjusted_rand_score(cat_labels, cls.labels_), 2)
    end = time.time()
    print("ARI : ", ARI, "duration : ", np.round(end - start, 1), "s")

    return ARI, X_tsne, cls.labels_


def TSNE_visu_fct(
    X_tsne,
    cat_labels,
    cluster_labels,
    categories,
    ARI,
    xlim=None,
    ylim=None,
    legend_loc="best",
    show_ticks=False,
    legend=True,
):
    """
    Function to visualize TSNE by real categories and clusters.

    Positional arguments :
    ----------------------

    X_tsne : 2darray
        embedding of the data in 2-dimensional space

    cat_labels : 1darray or list
        real category labels

    cluster_labels : 1darray or list
        cluster labels

    categories : list
        list of categories

    ARI : float
        ARI score between real categories and clusters

    optional arguments :
    --------------------

    xlim, ylim : tuples of floats
        limits of the x- and y-axis

    legend_loc : str or pair of floats, default='best'
        The location of the legend.

    show_ticks : bool, default=False
        whether to show ticks or not.

    legend : bool, default=True
        whether to show legends or not.
    """
    fig = plt.figure(figsize=(15, 6))

    # Plot t-SNE data labeled by real categories
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cat_labels, cmap="Set1")
    if legend:
        ax.legend(
            handles=scatter.legend_elements()[0],
            labels=categories,
            loc=legend_loc,
            title="Categories",
        )
    ax.set_title("Real categories", fontsize=20, fontweight="bold", pad=15)
    ax.set_xlabel("X_tsne1", fontsize=16, fontweight="normal")
    ax.set_ylabel("X_tsne2", fontsize=16, fontweight="normal")

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if show_ticks == False:
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot t-SNE data labeled by clusters
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap="Set1")

    cls_labels = [f"Cluster {i}" for i in set(cluster_labels)]
    if legend:
        ax.legend(
            handles=scatter.legend_elements()[0],
            labels=cls_labels,
            loc=legend_loc,
            title="Clusters",
        )
    ax.set_title("Clusters", fontsize=20, fontweight="bold", pad=15)
    ax.set_xlabel("X_tsne1", fontsize=16, fontweight="normal")
    ax.set_ylabel("X_tsne2", fontsize=16, fontweight="normal")

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if show_ticks == False:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    print("ARI : ", ARI)


def switch(cluster_labels, i, j):
    """Function that switches cluster labels"""
    store = 2 * (i + j) + 5
    cluster_labels[cluster_labels == i] = store
    cluster_labels[cluster_labels == j] = i
    cluster_labels[cluster_labels == store] = j
