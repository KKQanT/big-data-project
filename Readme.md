# Parallelized Clustering Algorithms with PySpark

This project implements three clustering algorithms - KMeans, KMode, KPrototype, and DBSCAN - using both local and global parallelized approach with PySpark. The algorithms are implemented separately and organized into different subdirectories within the repository.

## Clustering Algorithms Implemented

1. **KMeans**: KMeans is a popular clustering algorithm that partitions data into 'k' clusters based on similarity of features.

2. [**KMode**](https://projects.cs.nott.ac.uk/ppxpj2/big-data-project/-/tree/main/kmode?ref_type=heads): KMode is a clustering algorithm specifically designed for categorical data, where clusters are formed based on the mode of categories. Both local and global implementations are provided in this directory, along with an experimental notebook demonstrating how the size-up and num partition affect runtime. There are also notebooks that includes a step-by-step explanation of implementation.

3. **KPrototype**: KPrototype extends KMeans to handle both numerical and categorical data, allowing for more versatile clustering in mixed data types.

4. **DBSCAN**: DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together data points that are closely packed, while marking outliers as noise.


## Getting Started

To get started with using these clustering algorithms:

1. Navigate to the respective subdirectory for the algorithm you're interested in.
2. Follow the instructions provided in the README file within each directory for installation, usage, and examples.
3. Ensure you have PySpark installed and configured to run the algorithms in a distributed environment.