# Rocksdb Install

详细的安装方式，请参考[rocksdb官方安装说明]([rocksdb/INSTALL.md at master · facebook/rocksdb · GitHub](https://github.com/facebook/rocksdb/blob/master/INSTALL.md))。下面提供一些简单易上手的安装脚本。

## MacOS X

```bash
# 如果brew没有安装
# 请参考brew的官网：https://brew.sh/
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

# 线上环境不要用make/make all
# 因为会编译出debug模型
# 建议线上环境make static_lib/make shared_lib
# 具体参考官方文档
make shared_lib && make install-shared

# 添加到系统路径
cat >>/etc/profile <<EOF
export CPLUS_INCLUDE_PATH=\$CPLUS_INCLUDE_PATH:/usr/local/rocksdb/include/
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/rocksdb/lib64/
export LIBRARY_PATH=\$LIBRARY_PATH:/usr/local/rocksdb/lib64/
EOF
source /etc/profile
```