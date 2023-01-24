# damo-embedding
大规模稀疏模型训练的embedding部分的c++实现

## pyEmbedding库

### 安装
1. 依赖[rocksdb](https://github.com/facebook/rocksdb/)库
2. 安装命令: python3 setup.py install

### 使用方法
每一个embedding要单独设置，在代码里面key使用多个embedding，互相不干扰。