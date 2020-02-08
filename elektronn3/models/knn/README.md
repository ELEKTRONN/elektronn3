# NanoFLANN wrapper

## Compiling the K-nearest neighbors library

This directory was copied from https://github.com/aboulch/ConvPoint. The ```nearest_neighbors``` directory contains 
a very small wrapper for [NanoFLANN](https://github.com/jlblancoc/nanoflann) with OpenMP.
To compile the module:
```
cd knn
python setup.py install --home="."
```