
## Modules

Counting Bloom Filter

The purpose of the Counting Bloom Filter is to filter low-frequency features. For more details, please refer to [Counting Bloom Filter](CBF.md).

### Initializer

When user lookup keys from the embedding, if a key not exists, we use a specific initializer to initialize its weight, and then send the weight to the user, also save the weight to rocksdb. For more detail, please refer to [Initializer](Initializer.md).

### Optimizer

There are many different optimizers, user can pick one of them, and use this optimizer to apply gradients. For more detail, please refer to [Optimizer](Optimizer.md).

### Storage

The storage is based on rocksdb, it supports TTL(Time To Live). When creating a embedding object, storage object is necessary. Also, we support dump data to binary file for online serving. For more detail, please refer to [Storage](Storage.md).

### Embedding

This is the most important module in this project. When creating an embedding object, users need fill in 5 arguments listed below:

- **storage**: damo.PyStorage type
  
- **optimizer**: damo.PyOptimizer type
  
- **initializer**: damo.PyInitializer type
  
- **dimension**: int type, dim of embedding
  
- **group**: int type, [0, 256), defaul: 0
  

Embedding moule has two member functions: lookup and apply_gradients, both have no return values. For more detail, please refer to [Embedding](Embedding.md).
