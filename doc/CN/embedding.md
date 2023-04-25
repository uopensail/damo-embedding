# Embeddings

## Embedding类

## 初始化参数

1. int: 向量的维度

2. `std::shared_ptr<rocksdb::DBWithTTL>`: 用来存储向量的值和一些优化算子的中间值

3. `std::shared_ptr<Optimizer>`: 优化算子

4. `std::shared_ptr<Initializer>`: 初始化算子

5. int: 最少更新次数

## 成员函数

1. lookup: 查询embedding，如果命中或更新次数低于预设值，就返回0向量。以下是调用的参数：
   
   1. keys: 查询的key的向量
   
   2. len: 查询的key的长度
   
   3. data: 返回的浮点数的向量
   
   4. n: 返回的浮点数的向量长度

2. apply_gradients: 更新梯度

3. update: 内部优化算子处理

4. create: 内存初始化

## group的说明

在该模块里面key是用无符号的64为整型来表示，其中前8位(key >> 56)设置group，用来表示key的group，后56位($key \& (2^{56}-1)$)表示的key的值。这样不同的group可以设置不同的优化算子，初始化算子，维度等。

## storage

这里采用的是rocksdb来做存储，因为rocksdb本身支持了TTL，所以会定期把过期的key进行删除。另外在embedding