
## Install

### Swig And Numpy
When using damo-embedding,it is Not necessary to install SWIG. It is for development.

[Swig And Numpy](Swig&NumPy.md)

Use `swig -python -c++ -Wall -py3 damo.i` to regenerate the `damo_wrap.cxx` and `damo.py`


### RocksDB
[RocksDB](RocksDB.md)

When make rocksdb, must add these:

`EXTRA_CXXFLAGS=-fPIC EXTRA_CFLAGS=-fPIC USE_RTTI=1 DEBUG_LEVEL=0`

### Python3
This is python3 tool, [Python3](Python3.md) Is required. 

NumPy is needed, NumPy's include and lib path should add to system path, please refer to [Swig&NumPy.md](Swig&NumPy.md).

### install
```bash
python setup.py install