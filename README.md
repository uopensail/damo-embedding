# Damo-Embedding

[中文文档](doc/CN/readme_CN.md)

This project is mainly aimed at the model training scenario of small companies, because small companies may be limited in machine resources, and it is not easy to apply for large memory machines or distributed clusters. In addition, most small companies do not need distributed training when training machine learning/deep learning models. On the one hand, because small companies do not have enough data to train distributed large models. On the other hand, training distributed model is a relatively complex project, with high requirements for engineers, and the cost of machines is also high. However, if stand-alone training is used, Out-Of-Memory (OOM) and Out-Of-Vocabulary (OOV) problems often arise. Damo-Embedding is a project designed to solve these problems.

## Out-Of-Memory(OOM)

When using the machine learning framework (TensorFlow/Pytorch) to train the model, creating a new embedding is usually necessary to specify the dimension and size in advance. And also, their implementations are based on memory. If the embedding size is too large, there will be no enough memory. So why do you need such a large Embedding? Because in some scenarios, especially in search, recommmend or ads scenarios, the number of users and materials is usually very large, and engineers will do some manual cross-features, which will lead to exponential expansion of the number of features.

## Out-Of-Vocabulary(OOV)

In the online training model, some new features often appear, such as new user ids, new material ids, etc., which have never appeared before. This will cause the problem of OOV.

## Solutions

The reason for the OOV problem is mainly because the embedding in the training framework is implemented in the form of an array. Once the feature id is out of range, the problem of OOV will appear. We use [rocksdb](https://rocksdb.org/) to store embedding, which naturally avoids the problems of OOV and OOM, because rocksdb uses KV storage, which is similar to hash table and its capacity is only limited by the local disk.


## Modules

Counting Bloom Filter

The purpose of the Counting Bloom Filter is to filter low-frequency features. For more details, please refer to [Counting Bloom Filter](./doc/EN/CBF.md).

### Initializer

When user lookup keys from the embedding, if a key not exists, we use a specific initializer to initialize its weight, and then send the weight to the user, also save the weight to rocksdb. For more detail, please refer to [Initializer](./doc/EN/Initializer.md).

### Optimizer

There are many different optimizers, user can pick one of them, and use this optimizer to apply gradients. For more detail, please refer to [Optimizer](./doc/EN/Optimizer.md).

### Storage

The storage is based on rocksdb, it supports TTL(Time To Live). When creating a embedding object, storage object is necessary. Also, we support dump data to binary file for online serving. For more detail, please refer to [Storage](./doc/EN/Storage.md).

### Embedding

This is the most important module in this project. When creating an embedding object, users need fill in 5 arguments listed below:

- **storage**: damo.PyStorage type
  
- **optimizer**: damo.PyOptimizer type
  
- **initializer**: damo.PyInitializer type
  
- **dimension**: int type, dim of embedding
  
- **group**: int type, [0, 256), defaul: 0
  

Embedding moule has two member functions: lookup and apply_gradients, both have no return values. For more detail, please refer to [Embedding](./doc/EN/Embedding.md).

## Install

### Swig And Numpy
When using damo-embedding,it is Not necessary to install SWIG. It is for development.

[Swig And Numpy](doc/EN/Swig&NumPy.md)

Use `swig -python -c++ damo.i` to regenerate the `damo_wrap.cxx` and `damo.py`

### RocksDB
[RocksDB](doc/EN/RocksDB.md)

### Python3
[Python3](doc/EN/Python3.md)

```bash
python setup.py install
```


## Q&A

1. [undefined reference to `typeinfo for rocksdb::AssociativeMergeOperator'](https://github.com/facebook/rocksdb/issues/3811)
