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
        stopping_distance=None,
        n_partitions=10,
        verbose=True,
    ) -> None:

        self.K = K

        self.pdf = pd.read_csv(csv_path)

        if feature_cols is not None:
            self.pdf = self.pdf[feature_cols]
        if cols_to_drop is not None:
            self.pdf = self.pdf.drop(columns=cols_to_drop)

        self.max_iterations = max_iterations
        self.stopping_distance = stopping_distance

        if self.stopping_distance is None:
            self.stopping_distance = len(self.pdf.columns) // 10

        self.spark = (
            SparkSession.builder.master("local[*]")
            .appName("K-Mode-global")
            .getOrCreate()
        )

        self.sc = self.spark.sparkContext

        self.df = self.spark.createDataFrame(self.pdf)
        self.df.cache()
        self.rdd = self.df.rdd.repartition(n_partitions)
        self.rdd = self.rdd.map(lambda row: self.to_numpy(row))
        self.rdd.cache()

        self.centroid = None

        self.verbose = verbose

    def init_centroid(self):
        for idx, row in enumerate(
            self.df.takeSample(withReplacement=False, num=self.K)
        ):
            if idx == 0:
                centroid = np.array([self.to_numpy(row)])  # shape (1, P)
            else:
                c_ = self.to_numpy(row)
                c_ = np.array([c_])  # shape (1, P)
                centroid = np.concatenate([centroid, c_], axis=0)

            centroid = self.sc.broadcast(centroid)  # shape (K, P)

    def parallelize_fit(self):

        if self.centroid is None:
            self.init_centroid()

        for iter in range(self.max_iterations):

            combined_counts_rdd = self.rdd.mapPartitions(self.merge_counts_within_partition)

            counted_elem_rdd = combined_counts_rdd.reduceByKey(
                lambda x, y: KModeGlobal.merge_count_elem_hash(x, y)
            )
            cluster_and_new_centroid_rdd = counted_elem_rdd.map(
                lambda x: (x[0], KModeGlobal.get_centroid(x[1]))
            )
            centroid_hash_form = cluster_and_new_centroid_rdd.collect()

            for idx in range(self.K):
                if idx == 0:
                    new_centroid = np.array([centroid_hash_form[idx][1]])
                else:
                    mode = np.array([centroid_hash_form[idx][1]])
                new_centroid = np.concatenate([new_centroid, mode])

            old_centroid = self.centroid.value.copy()
            self.centroid = self.sc.broadcast(new_centroid)

            distance = KModeGlobal.hamming_distance(old_centroid, new_centroid)

            print(
                "iteration : ",
                {iter + 1},
                " hamming distance between new and previous centroid:  ",
                distance,
            )

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
        return closest_cluster

    def get_closest_cluster_and_count(self, x):
        closest_cluster = self.get_closest_cluster(x)

        P = len(x)

        count_elem_hash = {}
        for i in range(P):
            count_elem_hash[i] = {x[i]: 1}
        return (closest_cluster, count_elem_hash)

    @staticmethod
    def merge_count_elem_hash(count_elem_hash_A, count_elem_hash_B):
        for idx in count_elem_hash_A.keys():
            for key in count_elem_hash_B[idx].keys():
                if key in count_elem_hash_A[idx]:
                    count_elem_hash_A[idx][key] += count_elem_hash_B[idx][key]
                else:
                    count_elem_hash_A[idx][key] = 1
        return count_elem_hash_A

    def predict(self, x):
        return self.get_closest_cluster(x)

    @staticmethod
    def to_numpy(row):
        a = list(row.asDict().values())
        return np.array(a, dtype="<U22")

    @staticmethod
    def hamming_distance(x1, x2):
        return np.count_nonzero(x1 != x2)

    @staticmethod
    def get_centroid(count_elem_hash):
        P = len(count_elem_hash)
        centroid = np.full((P,), "", dtype="<U22")
        for idx, count_hash in count_elem_hash.items():
            mode = KModeGlobal.get_mode(count_hash)
            centroid[idx] = mode
        return centroid
    
    @staticmethod
    def get_mode(count_hash):
        mode = ""
        highest_count = 0
        for value, count in count_hash.items():
            if count > highest_count:
                mode = value
                highest_count = count
        return mode

    def merge_counts_within_partition(self, iterator):
        combined_counts = {}
        for x in iterator:
            cluster, count_hash = self.get_closest_cluster_and_count(x)
            if cluster in combined_counts:
                combined_counts[cluster] = self.merge_count_elem_hash(combined_counts[cluster], count_hash)
            else:
                combined_counts[cluster] = count_hash
        yield from combined_counts.items()