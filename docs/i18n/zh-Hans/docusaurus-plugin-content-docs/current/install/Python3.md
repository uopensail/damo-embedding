# Python3 Install

## MacOS

It is Recommended to install anaconda, refer to [anaconda's website](https://www.anaconda.com/)

## Linux

If you want to use anaconda,refer to [anaconda's website](https://www.anaconda.com/)

Below is python3.7 install script on centos7.

```bash
#!/bin/sh

# install sqlite
cd /tmp

yum install -y git wget
yum install -y libffi-devel ca-certificates
yum install -y openssl openssl-devel gcc-c++

wget https://www.sqlite.com/2021/sqlite-autoconf-3360000.tar.gz 
tar -xvf sqlite-autoconf-3360000.tar.gz
cd sqlite-autoconf-3360000 
./configure && make && make install 

cd /tmp
wget https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tgz
tar -zxvf Python-3.7.0.tgz 
cd Python-3.7.0
./configure prefix=/usr/local/python3 
make && make install
cp Lib/configparser.py /usr/local/python3/lib/python3.7/ConfigParser.py 
ln -s /usr/local/python3/bin/python3.7 /usr/bin/python3.7 
ln -s /usr/local/python3/bin/pip3.7 /usr/bin/pip3.7
pip3.7 install --upgrade pip 
```