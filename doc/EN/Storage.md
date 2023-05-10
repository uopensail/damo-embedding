# Storage

We use rocksdb to save the data. You should create the storage object fisrt, then you can create embedding objects.

When creating storage object, you should input data-dir and ttl. Data-dir is the path to save data. Ttl is time to live, which is supported by rocksdb.

## condition

conditions are wrap in parameter object.

### expire_days

If the last update time of the key is less then expire_days ago, this key is ignored.

configure parameters: 

**expire_days**: int type



### group

If the group is setted (0 <= group < 256),  we will only dump keys have the same group.

configure parameters:

**group**: int type



### min_count

If the update number of key is less then min_count, this key is ignored.

configure parameters:

**min_count**: int type



## dump

When model training finishes, you may dump the weights of keys with some conditions.

### file format

First part stores the dim and count of each group, totally 256 groups. Second part stores all the key and weigh.

#### first part

| type   | size | length | description                       |
| ------ | ---- | ------ | --------------------------------- |
| int32  | 4bit | 256    | dim of 256 group, default 0       |
| size_t | 8bit | 256    | key count of 256 group, default 0 |

#### second part

| type      | size | length            | description       |
| --------- | ---- | ----------------- | ----------------- |
| u_int64_t | 8bit | 1                 | key value         |
| int32     | 4bit | 1                 | group of the key  |
| float     | 4bit | dim of this group | weight of the key |

## checkpoint

When model training finishes, you may do the checkout of all the keys.

### file format

First part stores the count keys. Second part stores all the keys and values.

#### first part

| type   | size | length | description                       |
| ------ | ---- | ------ | --------------------------------- |
| size_t  | 8bit | 1    | key count     |

#### second part

| type      | size | length            | description       |
| --------- | ---- | ----------------- | ----------------- |
| size_t | 8bit | 1                 | key length        |
| byte   | 1bit | key length        | key data          |
| size_t | 8bit | 1                 | value length      |
| byte   | 1bit | value length      | value data        |


## Example

```python
import damo

# first param: data dir
# second param: ttl second
storage = damo.PyStorage("/tmp/data_dir", 86400*100)


cond = damo.Parameters()
cond.insert("expire_days", 100)
cond.insert("min_count", 3)
cond.insert("group", 0)

storage.dump("/tmp/weight.dat", cond)

storage.checkpoint("/tmp/checkpoint")

storage.load_from_checkpoint("/tmp/checkpoint")

```

## Extract Weights
```python
import struct
import numpy as np

path = "./data"

weight_dict = [{} for _ in range(256)]
with open(path, "rb") as f:
    data_for_dim = f.read(256 * 4)
    group_dims = struct.unpack("@256i", data_for_dim)
    print(group_dims)
    data_for_count = f.read(256 * 8)
    group_counts = struct.unpack("@256Q", data_for_count)
    print(group_counts)

    def get_weight(key):
        data_for_group = f.read(4)
        group = struct.unpack("@I", data_for_group)[0]
        key_dim = group_dims[group]
        data_for_weight = f.read(4 * key_dim)
        weight = struct.unpack(f"@{key_dim}f", data_for_weight)
        weight = np.array(weight, dtype=np.float32)
        weight_dict[group][key] = weight
        print(group, key, weight)

    data_for_key = f.read(8)
    while data_for_key:
        key = struct.unpack("@Q", data_for_key)[0]
        get_weight(key)
        data_for_key = f.read(8)
```