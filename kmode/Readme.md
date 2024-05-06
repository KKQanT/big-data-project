# KMode Clustering Algorithm

This directory contains both parellized local and global implementation of the KMode clustering algorithm using pyspark.

## Files Description

- `kmode_global.py`: This file contain global implementation of KMode as a python class object
  
- `kmode_local.py`: This file contain local implementation of KMode as a python class object. In each partition, it'll use sequential KMode object from `kmode_sequential.py` to learn on data in their parition.
- `kmode/experiments_notebook/K-Mode-global.ipynb` and `kmode/experiments_notebook/K-Mode-local.ipynb`: contains step-by-step implementation of parallelized KMode clustering algorithm with global and local approachs respectively. In this version, global approach is inefficent due to using numpy stack at the reduce phrase and move data to driver to calculate mode.
- `kmode/experiment_notebook/K-Mode-global2.ipynb` Efficient version of global approach where we calculate mode during the reduce phrase instead of moving the data outside
- `kmode/experiments_notebook/K-Mode-global-investigate-runtime.ipynb`: Notebook we used to investigate runtime of our implemented global approaches as we increased the size of the data increased