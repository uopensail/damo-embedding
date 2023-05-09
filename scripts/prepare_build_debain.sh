#!/bin/bash -x

apt-get update
apt-get install -y python-dev wget

# Build Rocksdb
apt-get install -y libgflags-dev
apt-get install -y libsnappy-dev zlib1g-dev libbz2-dev liblz4-dev libzstd-dev

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
