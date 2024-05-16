import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, hamming

class KPrototypes:
    """
    K-Prototypes Clustering Algorithm for mixed numerical and categorical data.
    """
    def __init__(self, data: pd.DataFrame, numerical_cols: list, categorical_cols: list, k_clusters: int):
        """
        Initialize KPrototypes instance.

        Args:
            data (pd.DataFrame): Input data containing both numerical and categorical columns.
            numerical_cols (list): List of numerical column names.
            categorical_cols (list): List of categorical column names.
            k_clusters (int): Number of clusters.
        """
        self.data = data
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.k_clusters = k_clusters
        self.centroids = self._initialize_centroids()

    def _initialize_centroids(self) -> pd.DataFrame:
        """
        Initialize centroids by randomly sampling from the data.

        Returns:
            pd.DataFrame: DataFrame containing centroids.
        """
        centroids = self.data.sample(n=self.k_clusters, random_state=42)
        centroids.reset_index(drop=True, inplace=True)
        return centroids

    def _assign_clusters(self) -> list:
        """
        Assign each data point to the nearest centroid.

        Returns:
            list: List of tuples containing index of data point and its assigned cluster.
        """
        clusters = []
        for idx, row in self.data.iterrows():
            min_dis = np.inf
            nearest_cluster = None
            for cen_idx, cen_row in self.centroids.iterrows():
                num_dis = euclidean(row[self.numerical_cols], cen_row[self.numerical_cols])
                cat_dis = hamming(row[self.categorical_cols], cen_row[self.categorical_cols])
                total_dis = num_dis + cat_dis
                if total_dis < min_dis:
                    min_dis = total_dis
                    nearest_cluster = cen_idx
            clusters.append((idx, nearest_cluster))
        return clusters

    def _update_centroids(self, clusters: list) -> None:
        """
        Update centroids based on assigned clusters.

        Args:
            clusters (list): List of tuples containing index of data point and its assigned cluster.
        """
        clusters_df = pd.DataFrame(clusters, columns=['row_index', 'cluster'])
        merged_df = pd.merge(clusters_df, self.data, left_on='row_index', right_index=True)
        self.centroids = merged_df.groupby('cluster').agg({
            col: 'mean' if col in self.numerical_cols else lambda x: x.mode().iloc[0] for col in self.data.columns
        })

    def fit(self) -> None:
        """
        Fit the model to the data by iteratively updating centroids until convergence.
        """
        prev_centroids = None
        step = 0
        while True and step < 20:  # Limiting to 20 steps to prevent infinite loops
            prev_centroids = self.centroids.copy()
            clusters = self._assign_clusters()
            self._update_centroids(clusters)
            step += 1
            if prev_centroids is not None:
                if self.centroids.equals(prev_centroids):
                    print('Convergence reached.')
                    break

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the cluster for each data point.

        Args:
            data (pd.DataFrame): New data for prediction.

        Returns:
            pd.DataFrame: DataFrame with a 'cluster' column indicating the predicted cluster for each data point.
        """
        clusters = []
        for idx, row in data.iterrows():
            min_dis = np.inf
            nearest_cluster = None
            for cen_idx, cen_row in self.centroids.iterrows():
                num_dis = euclidean(row[self.numerical_cols], cen_row[self.numerical_cols])
                cat_dis = hamming(row[self.categorical_cols], cen_row[self.categorical_cols])
                total_dis = num_dis + cat_dis
                if total_dis < min_dis:
                    min_dis = total_dis
                    nearest_cluster = cen_idx
            clusters.append(nearest_cluster)
        data['cluster'] = clusters
        return data

if __name__ == '__main__':
    
    data = pd.read_csv('your_data.csv')  
    numerical_cols = ['numerical_col1', 'numerical_col2']  
    categorical_cols = ['categorical_col1', 'categorical_col2']  
    k_clusters = 3
    k_proto = KPrototypes(data, numerical_cols, categorical_cols, k_clusters)
    k_proto.fit()
    clustering = k_proto.predict(data[:10])
    print(clustering)
