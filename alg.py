import numpy as np
from numpy.linalg import norm
from kmeansplus import k_init
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state

class Kmeans:
    '''Implementing Kmeans algorithm.'''

    def __init__(self, n_clusters, initial_clusters, order=2, max_iter=10000, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.order = order
        self.initial_clusters = initial_clusters

    def initializ_centroids(self, X):
        np.random.RandomState(self.random_state)
        # random_idx = np.random.permutation(X.shape[0])
        # centroids = X[random_idx[:self.n_clusters]]
        x_squared_norms = row_norms(X, squared=True)
        random = check_random_state(self.random_state)
        centroids = k_init(X, self.n_clusters, x_squared_norms, random)
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1, ord=self.order)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1, ord=2)
        return np.sum(np.square(distance))

    def fit(self, X):
        self.centroids = self.initial_clusters
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                print('Anzahl Iterationen bei ord={0}: {1}'.format(self.order, i))
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)

    def predict(self, X):
        distance = self.compute_distance(X, self.centroids)
        return self.find_closest_cluster(distance)
