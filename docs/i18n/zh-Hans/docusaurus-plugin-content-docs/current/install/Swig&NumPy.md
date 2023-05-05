# SWIG and NumPy

We use SWIG in this project to encapsulate C++ code into Python, SWIG installation can refer to this web page:

[How To Install Swig On MacOS, Linux And Windows](https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/)

## Grammar

The specific SWIG syntax can be referred to: [SWIG Official Documentation](https://www.swig.org/doc.html)

## NumPy

In this project, the numpy library is used. SWIG and NumPy need to be used together, so it needs to be provided [numyp.i](https://github.com/numpy/numpy/blob/main/tools/swig/numpy.i) file, refer to [numpy.i: a SWIG Interface File for NumPy](https://numpy.org/doc/stable/reference/swig.interface-file.html)。

### NumPy Add to System Path

```bash
# PYTHONPATH May Be Different

PYTHONPATH=/usr/local/python3/lib/python3.7
NUMPY_INCLUDE_PATH=$PYTHONPATH/site-packages/numpy/core/include
NUMPY_LIBRARY_PATH=$PYTHONPATH/site-packages/numpy/core/lib

export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$NUMPY_INCLUDE_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NUMPY_LIBRARY_PATH
export LIBRARY_PATH=$LIBRARY_PATH:NUMPY_LIBRARY_PATH
```