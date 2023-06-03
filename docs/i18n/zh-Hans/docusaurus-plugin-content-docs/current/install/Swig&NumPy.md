# SWIG and NumPy

SWIG在该项目中主要是将c++代码封装成Python的工具，具体的安装可以参考这篇网页：

[How To Install Swig On MacOS, Linux And Windows](https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/)

## 语法

具体的SWIG语法可以参考：[SWIG官方文档](https://www.swig.org/doc.html)

## NumPy

在该项目中，使用了numpy库，需要将SWIG和numpy结合起来使用，所以需要提供[numyp.i](https://github.com/numpy/numpy/blob/main/tools/swig/numpy.i)文件，具体使用方法参考[numpy.i: a SWIG Interface File for NumPy](https://numpy.org/doc/stable/reference/swig.interface-file.html)。
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