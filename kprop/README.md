# K-Prototype Global

This is an implementation of the K-Prototype clustering algorithm in PySpark. The K-Prototype algorithm is an extension of the K-Means and K-Mode algorithm that can handle both numerical and categorical data.

## Usage

1. Load your data into a Pandas DataFrame and pass it to the `df` variable. 
2. Define the labels for your data in the `labels` list. order: numerical labels first, followed by categorical labels.
3. Specify the index where the categorical labels start in the `categorical_labels_start_index` variable.
4. Run the code.

## Modifying Categorical Variable Importance

The relative importance given to categorical variables is controlled by the coefficient `0.2` next to the `hamming_distance` function. Increasing this value will give more importance to categorical variables during clustering.

## Using Other Distance Metrics

You can use other distance metrics for numerical and categorical data by modifying the `euclidean_distance` and `hamming_distance` functions, respectively. The new distance functions should follow the same format as the existing ones, returning a numpy array of distances between the input vector and the centroids.

## Dependencies

- PySpark
- Pandas
- NumPy