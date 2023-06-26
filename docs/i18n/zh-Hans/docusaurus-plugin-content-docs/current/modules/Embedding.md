
# Embedding

Embedding模块使用rocksdb来磁盘来存储Embedding的值, 采用KV的方式。 其中Key是特征hash的值(uint64类型), Value是Embedding对应的浮点数列表以及一些其他的值。

## Key

所有的特征都是经过离散化的, 用唯一的uint64的值进行表示。同一类特征用相同的特征组(group)来管理。在rocksdb中使用uint64这个值来进行查询。

## Group

这里会将特征分成不同的group, 不同的group可以设置不同的宽度, 不同的初始化算子和优化算子。

## Value

```c++
struct MetaData {
    int group; 
    int64_t key;
    int64_t update_time;
    int64_t update_num;
    float data[];
};
```

#### TTL

对于一些长时间没有更新的特征, 可以通过设置TTL的方式将其删除, 这个是rocksdb自身支持的功能。这样做也是为了降低模型的大小。

## Usage

### How to Create an Embedding

The arguments are listed below:

1. **storage**: damo.PyStorage type

2. **optimizer**: damo.PyOptimizer type

3. **initializer**: damo.PyInitializer type

4. **dimension**: int type, dim of embedding

5. **group**: int type, [0, 256), defaul: 0

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

1. keys: numpy.ndarray type, one dimension, dtype MUST BE np.uint64

2. weights: numpy.ndarray type, one dimension
   
   1. dtype MUST BE np.float32
   
   2. $size == embedding\_dimension * keys.shape[0]$
   
   We will store the result in this space. 

```python
import numpy as np

# example
n = 8
keys = np.zeros(n).astype(np.uint64)
for i in range(n):
    keys[i] = i+1

# array([1, 2, 3, 4, 5, 6, 7, 8], dtype=uint64)

weight = np.zeros(n*dimension).astype(np.float32)

embedding.lookup(keys, weight)

# IT IS Easy To Extract Each Key's Weight
tmp = weight.reshape((n, dimension))
weight_dict = {k: v for k,v in zip(keys, tmp)}
```

#### apply_gradients: push gradients to embedding

The arguments are listed below:

1. keys: same as lookup, numpy.ndarray type, one dimension, dtype MUST BE np.uint64

2. gradients: numpy.ndarray type, one dimension, dtype MUST BE np.float32 type, 
   
   1. dtype MUST BE np.float32
   
   2. $size == embedding\_dimension * keys.shape[0]$

```python
import numpy as np

gradients = np.random.random(n*dimension).astype(np.float32)

embedding.apply_gradients(keys, gradients)
```