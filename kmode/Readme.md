# KMode Clustering Algorithm

This directory contains both parallelized local and global implementations of the KMode clustering algorithm using PySpark.

## Files Description

### experiments_notebook
The notebooks in this folder contain both the Kmode global and local approaches. We also keep versions showing failures and mistakes to demonstrate what we've learned from this project.

**Please refer to `./experiments_notebook/K-Mode-global3.ipynb` and `./experiments_notebook/K-Mode-local2.ipynb` if you want to try out our implementation of the global and local approaches, respectively. The required packages to run our code are provided in requirements.txt.**

- `./experiments_notebook/K-Mode-local.ipynb`: The initial version of the local implementation where there was an issue with `np.apply_along_axis`, resulting in strings having only the first character, leading to incorrect centroids. This has been fixed in `./experiments_notebook/K-Mode-local2.ipynb`.

- `./experiments_notebook/K-Mode-global.ipynb`: The first version of the global implementation, where we simply mapped centroids in the mapping phase and grouped all the data in the reducing phase, then sent them back to the driver to calculate the mode. This was inefficient as it involved a lot of data movement and didn't utilize the reducing process well.

- `./experiments_notebook/K-Mode-global2.ipynb`: The second version of the global implementation, where we counted the number of occurrences of elements within the reducer phase to reduce data movement from the reducer phase to the driver. However, the data movement from the mapping phase to the reducing phase remained the same. We then implemented mini-reducers in the final version `./experiments_notebook/K-Mode-global3.ipynb`.

### Module
The final implementation has been rewritten in a class object style, but we suggest testing using notebooks for better visibility.

- `kmode_global.py`: This file contains the global implementation of KMode as a Python class object.

- `kmode_local.py`: This file contains the local implementation of KMode as a Python class object. In each partition, it uses a sequential KMode object from `kmode_sequential.py` to learn on data in their partition.
