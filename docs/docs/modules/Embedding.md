# Embedding

The Embedding module uses Rocksdb to store the values of Embedding, which is KV format. The Key of feature is int64_t type, the value is a list of floating point numbers and some other values.

## Key and Group

All features are discretization and represented by the unique int64_t value. We use group to represent the same type of features.Different group can have different optimizer, initializer and dimension.

## Value

```c++
struct MetaData {
    int32_t group; 
    int64_t key;  
    int64_t update_time;
    int64_t update_num;
    int32_t dim;
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

5. **group**: int type, [0, 256), defaul: 0

```python
import damo_embeddimg
optimizer = {...} # dict
initializer = {...} # dict
dimension = 16
embedding = damo_embeddimg.Embedding(dim=dimension, group=0, initializer=initializer, optimizer=optimizerp)
```

### Pull And Push for Embedding

There are two functions for embedding, both have no return values, which are listed below:

#### pull: pull weight from embedding

The arguments are listed below:

1. keys: numpy.ndarray type, one dimension, dtype MUST BE np.int64

2. weights: numpy.ndarray type, one dimension
   
   1. dtype MUST BE np.float32
   
   2. $size == embedding\_dimension * keys.shape[0]$
   
   We will store the result in this space. 

```python
import numpy as np
import damo

# example
n = 8
keys = np.zeros(n).astype(np.int64)
for i in range(n):
    keys[i] = i+1

group = 0

# array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int64)

weights = np.zeros(n*dimension).astype(np.float32)

damo.pull(group=group, keys=keys, weights=weights)

# IT IS Easy To Extract Each Key's Weight
tmp = weight.reshape((n, dimension))
weight_dict = {k: v for k,v in zip(keys, tmp)}
```

#### push: push gradients to embedding

The arguments are listed below:

1. keys: same as lookup, numpy.ndarray type, one dimension, dtype MUST BE np.int64

2. gradients: numpy.ndarray type, one dimension, dtype MUST BE np.float32 type, 
   
   1. dtype MUST BE np.float32
   
   2. $size == embedding\_dimension * keys.shape[0]$

```python
import numpy as np
import damo

group = 0

n = 8
keys = np.zeros(n).astype(np.int64)
for i in range(n):
    keys[i] = i+1

gradients = np.random.random(n*dimension).astype(np.float32)

damo.push(group=group, keys=keys, gradients=gradients)
```