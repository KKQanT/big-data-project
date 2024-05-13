### File Description

#### notebook
The provided code is an implementation of the K-means clustering algorithm using Apache Spark and NumPy. It consists of two main classes: KMeans and a helper class for data conversion.
- The KMeans class contains methods for initializing centroids, finding the closest centroid for a given data point, fitting the K-means model to the data, and transforming new      data points based on the learned centroids.
- The helper class partition_to_numpy_array is used to convert Apache Spark's RDD partition data into NumPy arrays, which are required for the K-means algorithm implementation.
- Additionally, there are two helper functions: get_optimal_kmeans_centroid and get_optimal_kmeans_centroid_with_dummy_key. These functions are used to obtain the optimal            centroids for each partition and to handle the key-value pair format required by Apache Spark's reduceByKey operation.

The notebook includes a section that iterates over different fractions of the input data and different numbers of partitions, performing the parallel K-means clustering and reporting the results.
It's important to note that for small datasets, the overhead of distributed computing may outweigh the benefits of parallelization. However, this implementation can serve as a foundation for handling larger datasets more efficiently in the future.
Note: Please install pyspark and ucimlrepo to run this code.
