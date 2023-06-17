# Counting Bloom Filter

## Why Counting Bloom Filter

The purpose of the Counting Bloom Filter(abbr. CBF) is to filter low-frequency features. 

Long-tail is a distinctive characteristic in internet scenario. When training machine learning/deep learning models, there will be too many long-tail features, its requency is quite low. They do harm to the convergence of the model: on the one hand, the low-frequency features are not fully trained, on the other hand, they also waste large storage resources and computing resources, so it is very necessary to remove low-frequency features.

If the model is trained offline, the engineer can preprocess the features, count the frequency of each feature, and then remove these low-frequency features. However, if it is an online model, it is not possible to preprocess the features. There are many schemes for processing sparse features, such as: feature frequency estimation based on Poisson distribution, dynamic adjustment of L1 regular filtering, etc.[1]. 

We provides a relatively straightforward way, using the CBF to record the number of feature's frequency. It should be noted that we uses `4bit`to store the number, which means that the maximum frequency is 15. Because we believe that the value of 15 can meet most of the needs.

Also, to avoid the problem of data loss. We use `mmap`,which maps the file to memory, to save data. If the model training finishes or crashes, data has already been saved to the disk. When the model training is restarted, you can reload the data from disk

There is a question on stackoverflow, [What updates mtime after writing to memory mapped files?](https://stackoverflow.com/questions/44815329/what-updates-mtime-after-writing-to-memory-mapped-files)

> When you `mmap` a file, you're basically sharing memory directly between your process and the kernel's page cache — the same cache that holds file data that's been read from disk, or is waiting to be written to disk. A page in the page cache that's different from what's on disk (because it's been written to) is referred to as "dirty".
> There is a kernel thread that scans for dirty pages and writes them back to disk, under the control of several parameters. One important one is `dirty_expire_centisecs`. If any of the pages for a file have been dirty for longer than `dirty_expire_centisecs` then all of the dirty pages for that file will get written out. The default value is 3000 centisecs (30 seconds).

Because mmap writes data to disk periodically, there is no need to create a new thread to write data to disk.



## Configuration

1. capacity: max capacity of CBF, default: $2^{28}$

2. count: filter count, default: 15

3. path: data path, default: /tmp/COUNTING_BLOOM_FILTER_DATA

4. fpr: false positive rate, default: 1e-3

5. reload: whether read data from disk file, defalut: true

## Example

```python
import damo

param = damo.Parameters()
param.insert("capacity", 1<<28) 
param.insert("count", 15)  
param.insert("path", "/tmp/cbf")
param.insert("reload", True)
param.insert("fpr", 0.001)
print(param.to_json())

filter = damo.PyFilter(param)

group = 1
key = 123456
for i in range(16):
    filter.add(group, key, 1)
    print(filter.check(group, key))
```

## Reference

[1][Ant Financial's core technology: real-time recommendation algorithm for tens of billions of features](https://developer.aliyun.com/article/714366)