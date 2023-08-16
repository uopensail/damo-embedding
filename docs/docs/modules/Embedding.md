# Embedding

The Embedding module uses Rocksdb to store the values of Embedding, which is KV format. The Key of feature is uint64_t type, the value is a list of floating point numbers and some other values.

## Key and Group

All features are discretization and represented by the unique uint64_t value. We use group to represent the same type of features.Different group can have different optimizer, initializer and dimension.

## Value

```c++
struct MetaData {
    int group; 
    int64_t key;  
    int64_t update_time;
    int64_t update_num;
    int dim;
    float data[];
};
```

## TTL

For some features that have not been updated for a long time, they can be deleted by setting TTL, which is supported by Rocksdb itself. This action can reduce the size of the model.

## Usage

### How to Create an Embedding

The arguments are listed below:

1. **storage**: damo.PyStorage type

2. **optimizer**: damo.PyOptimizer type

3. **initializer**: damo.PyInitializer type

4. **dimension**: int type, dim of embedding

5. **group**: int type, [0, 2^16), defaul: 0

```python
import damo
storage = damo.PyStorage(...)
optimizer = damo.PyOptimizer(...)
initializer = damo.PyInitializer(...)
dimension = 16
group = 1
embedding = damo.PyEmbedding(storage, optimizer, initializer, dimension, group)
```

### Member Functions of Embedding

There are two member functions of embedding, both have no return values, which are listed below:

#### lookup: pull weight from embedding

The arguments are listed below:

1. keys: numpy.ndarray type, one dimension, dtype MUST BE np.int64

2. weights: numpy.ndarray type, one dimension
   
   1. dtype MUST BE np.float32
   
   2. $size == embedding\_dimension * keys.shape[0]$
   
   We will store the result in this space. 

```python
import numpy as np

# example
n = 8
keys = np.zeros(n).astype(np.int64)
for i in range(n):
    keys[i] = i+1

# array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int64)

weight = np.zeros(n*dimension).astype(np.float32)

embedding.lookup(keys, weight)

# IT IS Easy To Extract Each Key's Weight
tmp = weight.reshape((n, dimension))
weight_dict = {k: v for k,v in zip(keys, tmp)}
```

#### apply_gradients: push gradients to embedding

The arguments are listed below:

1. keys: same as lookup, numpy.ndarray type, one dimension, dtype MUST BE np.int64

2. gradients: numpy.ndarray type, one dimension, dtype MUST BE np.float32 type, 
   
   1. dtype MUST BE np.float32
   
   2. $size == embedding\_dimension * keys.shape[0]$

```python
import numpy as np

gradients = np.random.random(n*dimension).astype(np.float32)

embedding.apply_gradients(keys, gradients)
```