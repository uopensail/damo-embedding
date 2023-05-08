#!/bin/bash -x

yum install -y python-devel wget
yum install -y git gflags-devel snappy-devel glog-devel zlib-devel \
    lz4-devel libzstd-devel gcc-c++ make autoreconf automake \
    libtool cmake 

# Build With Source
gcc -v
cd /tmp
wget https://github.com/facebook/rocksdb/archive/v6.4.6.tar.gz
tar -xvzf v6.4.6.tar.gz
cd rocksdb-6.4.6/

# online env NOT USE make/make all
# online env make static_lib/make shared_lib
# please refer to the official documentation for details
make shared_lib EXTRA_CXXFLAGS=-fPIC EXTRA_CFLAGS=-fPIC USE_RTTI=1 DEBUG_LEVEL=0 
make install-shared
