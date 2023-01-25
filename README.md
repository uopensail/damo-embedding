# damo-embedding
大规模稀疏模型训练的embedding部分的c++实现

## pyEmbedding库

### 安装
1. 依赖[rocksdb](https://github.com/facebook/rocksdb/)库
2. 利用swig生成代码
```shell
swig -python -c++ pyembedding.i
```
3. 安装命令: python3 setup.py install
备注:
numpy的include path要添加到系统路径


### 使用方法



