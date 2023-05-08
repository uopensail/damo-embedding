#!/bin/bash -x

apt-get update
apt-get install -y python-dev wget

# Build Rocksdb
apt-get install -y libgflags-dev
apt-get install -y libsnappy-dev zlib1g-dev libbz2-dev liblz4-dev libzstd-dev

# Build With Source

cd /tmp
wget https://github.com/facebook/rocksdb/archive/v8.1.1.tar.gz
tar -xvzf v8.1.1.tar.gz
cd rocksdb-8.1.1/
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/rocksdb ..

# online env NOT USE make/make all
# online env make static_lib/make shared_lib
# please refer to the official documentation for details
make shared_lib EXTRA_CXXFLAGS=-fPIC EXTRA_CFLAGS=-fPIC USE_RTTI=1 DEBUG_LEVEL=0 
make install-shared
