# swig

SWIG在该项目中主要是将c++代码封装成Python的工具，具体的安装可以参考这篇网页：

[How To Install Swig On MacOS, Linux And Windows](https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/)

## 语法

具体的SWIG语法可以参考：[SWIG官方文档](https://www.swig.org/doc.html)

## NumPy

在该项目中，使用了numpy库，需要将SWIG和numpy结合起来使用，所以需要提供[numyp.i](https://github.com/numpy/numpy/blob/main/tools/swig/numpy.i)文件，具体使用方法参考[numpy.i: a SWIG Interface File for NumPy](https://numpy.org/doc/stable/reference/swig.interface-file.html)。



### NumPy添加到系统路径

numpy的头文件一般在`$PYTHONPATH/site-packages/numpy/core/include`这个路径下