# Damo-Embedding安装

## 安装依赖

### Rocksdb安装

[Rocksdb安装脚本](RocksDB.md)

### Python3安装

[Python3安装脚本](Python3.md)

### Numpy安装

```bash
pip3.7 install numpy
```

一般来说numpy安装在了python的路径下`site-packages`文件中

```bash
PYTHONPATH=/usr/local/python3/lib/python3.7
NUMPY_INCLUDE_PATH=$PYTHONPATH/site-packages/numpy/core/include
NUMPY_LIBRARY_PATH=$PYTHONPATH/site-packages/numpy/core/lib

export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$NUMPY_INCLUDE_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NUMPY_LIBRARY_PATH
export LIBRARY_PATH=$LIBRARY_PATH:NUMPY_LIBRARY_PATH
```

## damo-enmbedding安装

```bash
python3.7 setup.py install
```