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
| uint64_t | 8bit | 1                 | key value         |
| int32     | 4bit | 1                 | group of the key  |
| float     | 4bit | dim of this group | weight of the key |

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

```