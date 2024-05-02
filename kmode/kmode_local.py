import pandas as pd
import gc
from pyspark.sql import SparkSession
import numpy as np
from collections import Counter
import random
from kmode_sequential import KMode

class KModeLocal:

    def __init__(
        self,
        csv_path,
        K=3,
        cols_to_drop=None,
        feature_cols=None,
        max_iterations=10,
        stopping_distance=1,
        n_partitions=100,
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

        self.spark = (
            SparkSession.builder.master("local[*]")
            .appName("K-Mode-local")
            .getOrCreate()
        )

        self.sc = self.spark.sparkContext

        self.df = self.spark.createDataFrame(self.pdf)
        self.df.cache()
        self.rdd = self.df.rdd.repartition(n_partitions)
        self.rdd.cache()

    def parallelized_fit(self):
        self.rdd = self.rdd.mapPartitions(KModeLocal.parition_to_numpy_array)
        self.rdd = self.rdd.map(self.get_optimal_kmode_centroid)
        all_centroids = self.rdd.reduce(lambda x, y: np.vstack((x, y)))
        final_kmode = KMode(self.K)
        final_kmode.fit(all_centroids)
        self.centroid = final_kmode.centroid.copy()


    @staticmethod
    def parition_to_numpy_array(instances):
        array = None
        for row in instances:
            if array is None:
                array = np.array(list(row.asDict().values()), dtype="<U22")
            else:
                array_row = np.array(list(row.asDict().values()), dtype="<U22")
                array = np.vstack((array, array_row))
        yield array
    
    def get_optimal_kmode_centroid(self, X):
        kmode = KMode(self.K)
        kmode.fit(X)
        return kmode.centroid


