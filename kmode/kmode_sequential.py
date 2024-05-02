import random
import numpy as np
from collections import Counter

class KMode:

    def __init__(self, K: int) -> None:
        self.K = K

    def init_centroid(self, X):
        col_len = X.shape[1]
        centroid = np.array([], dtype="<U22")
        for i in range(self.K):
            mode = []
            for col in range(col_len):
                rand_val = random.choice(np.unique(X[:, col]))
                mode.append(rand_val)
            if i == 0:
                centroid = np.array(mode, dtype="<U22")
                del mode
            else:
                mode = np.array(mode, dtype="<U22")
                centroid = np.vstack((centroid, mode))
        self.centroid = centroid
        return centroid

    def get_closest_centroid(self, x):
        min_hamming_distance = np.inf
        closest_cluster = 0
        for i, mode in enumerate(self.centroid):
            distance = self.hamming_distance(x, mode)
            if distance < min_hamming_distance:
                min_hamming_distance = distance
                closest_cluster = i
        return closest_cluster

    def fit(self, X, n_iters=100, stopping_criterion=1) -> None:
        self.init_centroid(X)
        self.clustered = np.zeros(X.shape[0])
        for iter in range(n_iters):
            loss = self.fit_one_step(X)
            print(f"iter: {iter} loss: {loss}")
            if loss < stopping_criterion:
                break

    def fit_one_step(self, X) -> int:
        self.prev_centroid = self.centroid.copy()
        for i, x in enumerate(X):
            closest_centroid = self.get_closest_centroid(x)
            self.clustered[i] = closest_centroid
        for i in range(self.K):
            idx = np.where(self.clustered == i)[0]
            if idx.shape[0] > 0:
                new_mode = self.get_mode_from_arr(X[idx])
                self.centroid[i] = new_mode
        return self.hamming_distance(self.centroid, self.prev_centroid)

    def transform(self, X_test):
        clustered = np.zeros(X_test.shape[0])
        for i, x in enumerate(X_test):
            closest_centroid = self.get_closest_centroid(x)
            clustered[i] = closest_centroid
        return clustered

    @staticmethod
    def hamming_distance(a, b) -> int:
        return np.count_nonzero(a != b)

    @staticmethod
    def get_mode_from_arr(arr):
        def get_mode_from_vec(vec):
            counted = Counter(vec)
            return counted.most_common(1)[0][0]

        return np.apply_along_axis(get_mode_from_vec, 0, arr)