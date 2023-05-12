# RocksDB Install

Refer to [rocksdb/INSTALL.md at master · facebook/rocksdb · GitHub](https://github.com/facebook/rocksdb/blob/master/INSTALL.md)。Below are some easy install commands.

## MacOS X

```bash
# if brew is not installed
# please refer to brew's website: https://brew.sh/
brew install rocksdb
```

## Linux - CentOS

```bash
#!/bin/sh
yum install -y git gflags-devel snappy-devel glog-devel zlib-devel \
    lz4-devel libzstd-devel gcc-c++ make autoreconf automake \
    libtool cmake

cd /tmp
wget https://github.com/facebook/rocksdb/archive/v6.4.6.tar.gz
tar -xvzf v6.4.6.tar.gz
cd rocksdb-6.4.6/
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/rocksdb ..

# online env NOT USE make/make all
# online env make static_lib/make shared_lib
# please refer to the official documentation for details
make shared_lib EXTRA_CXXFLAGS=-fPIC EXTRA_CFLAGS=-fPIC USE_RTTI=1 DEBUG_LEVEL=0 
make install-shared

# add to system path
cat >>/etc/profile <<EOF
export CPLUS_INCLUDE_PATH=\$CPLUS_INCLUDE_PATH:/usr/local/rocksdb/include/
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/rocksdb/lib64/
export LIBRARY_PATH=\$LIBRARY_PATH:/usr/local/rocksdb/lib64/
EOF
source /etc/profile
```