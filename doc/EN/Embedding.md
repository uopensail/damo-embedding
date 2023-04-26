# Embedding

The Embedding module uses Rocksdb to store the values of Embedding, which is KV format. The Key of feature is u_int64_t type, the value is a list of floating point numbers and some other values.

## Key and Group

All features are discretization and represented by the unique u_int64_t value. We use group to represent the same type of features.Different group can have different optimizer, initializer and dimension.

#### Value

```c++
struct MetaData {
    int group; 
    u_int64_t key;
    u_int64_t update_time;
    u_int64_t update_num;
    float data[];
};
```

#### TTL

For some features that have not been updated for a long time, they can be deleted by setting TTL, which is supported by Rocksdb itself. This action can reduce the size of the model.

