## Modules

### Initializer

When user lookup keys from the embedding, if a key not exists, we use a specific initializer to initialize its weight, and then send the weight to the user, also save the weight to rocksdb. For more detail, please refer to [Initializer](Initializer.md).

### Optimizer

There are many different optimizers, user can pick one of them, and use this optimizer to apply gradients. For more detail, please refer to [Optimizer](Optimizer.md).

### Embedding

This is the most important module in this project. When creating an embedding object, users need fill in 3 arguments listed below:
  
- **dim**: int type, dim of embedding

- **optimizer**: dict type
  
- **initializer**: dict type
  

Embedding moule has two member functions: lookup and apply_gradients, both have no return values. For more detail, please refer to [Embedding](Embedding.md).