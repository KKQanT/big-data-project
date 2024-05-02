import pandas as pd
import gc
from pyspark.sql import SparkSession
import numpy as np
from collections import Counter
import random


class KModeGlobal:

    def __init__(
        self,
        csv_path,
        K=3,
        cols_to_drop=None,
        feature_cols=None,
        max_iterations=10,
        stopping_distance=1,
        n_partitions=10,
        verbose=True
    ) -> None:
        
        self.K = K

        self.pdf = pd.read_csv(csv_path)

        if feature_cols is not None:
            self.pdf = self.pdf[feature_cols]
        if cols_to_drop is not None:
            self.pdf = self.pdf.drop(columns=cols_to_drop)

        self.max_iterations = max_iterations
        self.stopping_distance = stopping_distance

        self.spark = (
            SparkSession.builder.master("local[*]")
            .appName("K-Mode-global")
            .getOrCreate()
        )

        self.sc = self.spark.sparkContext

        self.df = self.spark.createDataFrame(self.pdf)
        self.df.cache()
        self.rdd = self.df.rdd.repartition(n_partitions)
        self.rdd.cache()

        self.generate_unique_value_of_all_features()

        self.centroid = None

        self.verbose = verbose

    def generate_unique_value_of_all_features(self):
        columns = self.df.columns
        self.unique_values_dict = {}
        for col in columns:
            unique_val_objs = self.df.select(col).distinct().collect()
            unique_val_list = [row[col] for row in unique_val_objs]
            self.unique_values_dict[col] = unique_val_list
            del unique_val_list
            gc.collect()

    def init_centroid(self):
        for i, col in enumerate(self.df.columns):
            unique_values = self.unique_values_dict[col]
            ramdom_vals = random.choices(unique_values, k=self.K)
            if i == 0:
                centroid = np.array(ramdom_vals).reshape(-1, 1).astype("str")
            else:
                ramdom_vals = np.array(ramdom_vals).reshape(-1, 1).astype("str")
                centroid = np.hstack((centroid, ramdom_vals))

            self.centroid = self.sc.broadcast(centroid)

    def parallelize_fit(self):
        
        if self.centroid is None:
            self.init_centroid()

        for iter in range(self.max_iterations):
            clustered = self.rdd.map(
                lambda x: self.get_closest_cluster(x)
            )  # -> (k, v) = (cluster_i, X)
            group_by_clustered = clustered.reduceByKey(lambda x, y: np.vstack((x, y)))
            centroid_rdd = group_by_clustered.map(
                lambda x: (x[0], KModeGlobal.get_mode_from_arr(x[1]))
            )
            centroid_list = centroid_rdd.collect()

            new_centroid = self.centroid.value.copy()
            for (i, arr) in centroid_list:
                new_centroid[i] = arr

            old_centroid = self.centroid.value.copy()
            self.centroid = self.sc.broadcast(new_centroid)

            distance = KModeGlobal.hamming_distance(old_centroid, new_centroid)

            if self.verbose:
                print('iteration : ', {iter+1}, " hamming distance between new and previous centroid:  ", distance)

            if distance <= self.stop_distance:
                break
    
    def get_closest_cluster(self, x):
        min_hamming_distance = np.inf
        closest_cluster = 0
        for i, mode in enumerate(self.centroid.value):
            distance = KModeGlobal.hamming_distance(x, mode)
            if distance < min_hamming_distance:
                min_hamming_distance = distance
                closest_cluster = i
        return (closest_cluster, x)
    
    def predict(self, x):
        return self.get_closest_cluster(x)[0]

    @staticmethod
    def to_numpy(row):
        a = list(row.asDict().values())
        return np.array(a, dtype="<U22")

    @staticmethod
    def hamming_distance(x1, x2):
        return np.count_nonzero(x1 != x2)

    @staticmethod
    def get_mode_from_vec(vec):
        counted = Counter(vec)
        return counted.most_common(1)[0][0]

    @staticmethod
    def get_mode_from_arr(arr):
        return np.apply_along_axis(KModeGlobal.get_mode_from_vec, 0, arr)
