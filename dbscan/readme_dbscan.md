# DBSCAN Implementation

This is an implementation of the DBSCAN algorithm using Apache Spark. It is tailored to handle geospatial data, which is partitioned and processed by DBSCAN clustering on each partition. The implementation also includes a strategy to merge these local clusters into global clusters efficiently.

## Dataset

The geospatial data used in this project can be found at the following link:

[Geospatial Data on Kaggle](https://www.kaggle.com/datasets/jeniannamathew/geodata)

## Running the Implementation

To see the implementation in action, you can run the Kaggle notebook where the data is pre-loaded.

[DBSCAN Implementation on Kaggle](https://www.kaggle.com/code/jeniannamathew/dbscan)

Alternatively, you can download the dataset from the provided link, set the correct path to the root directory of the data, and run the code in your local environment.

## Dependencies

To run this project, you will need the following Python packages installed:

- PySpark
- Pandas
- NumPy
