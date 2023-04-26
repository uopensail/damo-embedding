# Damo-Embedding

该项目主要针对的是小公司的模型训练场景, 因为小公司在机器资源方面可能比较受限, 不太容易申请大内存的机器以及分布式的集群。另外, 大部分的小公司在训练机器学习/深度学习模型的时候, 其实是不需要分布式训练的。一方面因为小公司的数据量不足以训练分布式的大模型, 另一方面分布式模型训练是一个比较复杂的工程, 对工程师的要求较高, 而且服务器的成本也是偏高。但是, 如果采用单机训练的话, 往往会出现Out-Of-Memory(OOM)和Out-Of-Vocabulary(OOV)的问题。Damo-Embedding就是用来解决这些问题的项目。

## Out-Of-Memory(OOM)

在使用机器学习框架(TensorFlow/Pytorch)训练模型的时候, 新建Embedding的时候, 通常需要提前指定维度和长度。它们的实现都是基于内存的, 如果申请的Embedding过大, 就会出现内存不够用的情形。那么为什么会需要那么大的Embedding呢？因为在一些实际业务中, 尤其是搜推广业务, 通常是用户和物料的数量都很大, 另外算法工程师会做一些手工交叉的特征, 这样会导致特征数量指数级膨胀。

## Out-Of-Vocabulary(OOV)

在线训练的模型中, 往往会出现一些新的特征, 比如新的用户id, 新的物料id等, 这些特征之前从未出现过。这样就会出现OOV的问题。

## Solutions

之所以会出现OOV的问题, 主要是因为训练框架中的Embedding采用的是数组的方式来实现的。一旦特征id超出范围就会出现OOV的问题。开发者采用[rocksdb](https://rocksdb.org/)来存放Embedding, 就天然避免了OOV和OOM的问题, 因为rocksdb采用的是KV存储, 类似于hash table且容量大小仅仅受本地磁盘的限制。

## Modules

1. Embedding: 用rocksdb来存储embedding权重
2. [Counting Bloom Filter](https://en.wikipedia.org/wiki/Counting_Bloom_filter): 用来过滤低频次特征, 降低模型复杂度
3. Initializer: 初始化算子
4. Optimizer: 优化算子

### Embedding

Embedding模块使用rocksdb来磁盘来存储Embedding的值, 采用KV的方式。 其中Key是特征hash的值(uint64类型), Value是Embedding对应的浮点数列表以及一些其他的值。

#### Key

所有的特征都是经过离散化的, 用唯一的uint64的值进行表示。同一类特征用相同的特征组(group)来管理。在rocksdb中使用uint64这个值来进行查询。

#### Group

这里会将特征分成不同的group, 不同的group可以设置不同的宽度, 不同的初始化算子和优化算子。


#### Value

```c++
struct MetaData {
    int group;                      // 特征组
    u_int64_t key;                  // 主键
    u_int64_t update_time;          // 更新时间
    u_int64_t update_num;           // 更新此次
    float data[];                    // 实际数据
};
```

#### TTL

对于一些长时间没有更新的特征, 可以通过设置TTL的方式将其删除, 这个是rocksdb自身支持的功能。这样做也是为了降低模型的大小。

### Counting Bloom Filter (abbr. CBF)

CBF的作用是用来过滤低频次特征。互联网的业务一般都呈现长尾的特点, 在训练机器学习/深度学习模型的时候, 长尾特征会非常多，其中大部分的长尾特征出现的频次非常低。它们对模型的收敛产生了不小的干扰: 一方面特征频次低导致这些特征无法充分训练, 另一方面它们对存储资源和计算资源消耗也非常大, 所以去掉低频次的特征就显得非常有必要。

如果是离线训练模型, 算法工程师可以进行特征的预处理, 统计特征出现的频次, 然后把这些低频次的特征去掉。但是如果是在线模型, 就无法进行特征的预处理。对稀疏特征进行处理有很多方案, 例如: 基于泊松分布的特征频次估计, 动态调整L1正则过滤等<sup>[1]</sup>。该项目提供了一种相对比较直接的方式, 利用CBF来记录特征出现的次数实现特征频次过滤。需要说明的是, 项目中采用`4bit`来存放每一个数据, 也就是说最大的记录数就是15。因为开发者认为15这个值已经可以满足绝大部分需求了。

另外, 为了避免数据丢失的问题。开发者使用mmap的技术将内存和文件进行了映射。当这次模型训练完结或中途出现崩溃, 模型训练重启的时候, 可以通过加载文件的方式把数据恢复起来。

参考stackoverflow上的一个问题[What updates mtime after writing to memory mapped files?](https://stackoverflow.com/questions/44815329/what-updates-mtime-after-writing-to-memory-mapped-files)

> When you `mmap` a file, you're basically sharing memory directly between your process and the kernel's page cache — the same cache that holds file data that's been read from disk, or is waiting to be written to disk. A page in the page cache that's different from what's on disk (because it's been written to) is referred to as "dirty".
> There is a kernel thread that scans for dirty pages and writes them back to disk, under the control of several parameters. One important one is `dirty_expire_centisecs`. If any of the pages for a file have been dirty for longer than `dirty_expire_centisecs` then all of the dirty pages for that file will get written out. The default value is 3000 centisecs (30 seconds).

注：

1. 因为mmap会将数据定期写到磁盘，就不需要另起线程写到磁盘。

2. 原本CBF是嵌入到Embedding里面的，使用者配置一下就能实现过滤。后来经过考虑，把CBF作为一个功能点拆出来，使用者显式的去查询和插入，手动过滤。

#### Configuration
使用CBF需要配置如下的一些参数：
1. capacity: 最大容量, default: 2^28
2. count: 过滤次数, default: 15
3. path: 保存路径, default: /tmp/COUNTING_BLOOM_FILTER_DATA
4. fpr: 假阳率, default: 1e-3
5. reload: 是否加载数据文件, default: true

### Initializer

#### Zeros

0初始化
需要配置如下参数:
1. name: "zeros"

#### Ones

1初始化
需要配置如下参数:
1. name: "ones"

#### RandomUniform

均匀分布, 需要配置如下的一些参数:
1. name: "random_uniform"
2. min: 下限, 浮点数, default: -1.0
3. max: 上限, 浮点数, default: 1.0

#### RandomNormal

随机正态分布, 需要配置如下的一些参数:
1. name: "random_normal"
2. mean: 均值, 浮点数, default: 0.0
3. stddev: 标准差, 浮点数, default: 1.0

#### TruncateNormal

随机正态分布, 且2倍标准差外的数据丢弃重新生成 需要配置如下的一些参数:
1. name: "truncate_normal"
2. mean: 均值, 浮点数, default: 0.0
3. stddev: 标准差, 浮点数, default: 1.0

### Optimizer

#### SGD

[SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD)需要配置如下的一些参数:

1. $\gamma$: 学习率, default: 1e-3, 配置名称: gamma
2. $\lambda$: 权重衰减的系数, default: 0, 配置名称: lambda

#### FTRL

[FTRL](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/37013.pdf)需要配置如下的一些参数:

1. $\alpha$: 学习率, default: 5e-3, 配置名称: gamma
2. $\beta$: $\beta$参数, default: 0.0, 配置名称: beta
3. $\lambda_1$: L1正则参数, default: 0.0, 配置名称: lambda1
4. $\lambda_2$: L2正则参数, default: 0.0, 配置名称: lambda2

#### Adagrad

[Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad)需要配置如下的一些参数:

1. $\gamma$: 学习率, default: 1e-2, 配置名称: gamma
2. $\lambda$: 权重衰减的系数, default: 0.0, 配置名称: lambda
3. $\eta$: 学习率衰减系数, default: 0.0, 配置名称: eta
4. $\epsilon$: 最小误差项, default: 1e-10, 配置名称: epsilon

#### Adam

[Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)需要配置如下的一些参数(不支持amsgrad):

1. $\gamma$: 学习率, default: 1e-3, 配置名称: gamma
2. $\beta_1$: 梯度的移动均值系数, default: 0.9, 配置名称: beta1
3. $\beta_2$: 梯度平方的移动均值系数, default: 0.999, 配置名称: beta2
4. $\lambda$: 权重衰减的系数, default: 0.0, 配置名称: lambda
5. $\epsilon$: 最小误差项, default: 1e-8, 配置名称: epsilon

#### AdamW

[AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW)需要配置如下的一些参数(不支持amsgrad):

1. $\gamma$: 学习率, default: 1e-3, 配置名称: gamma
2. $\beta_1$: 梯度的移动均值系数, default: 0.9, 配置名称: beta1
3. $\beta_2$: 梯度平方的移动均值系数, default: 0.999, 配置名称: beta2
4. $\lambda$: 权重衰减的系数, default: 1e-3, 配置名称: lambda
5. $\epsilon$: 最小误差项, default: 1e-8, 配置名称: epsilon

#### Lion

[Lion](https://arxiv.org/abs/2302.06675)需要配置如下的一些参数:

1. $\eta$: 学习率, default: 3e-4, 配置名称: eta
2. $\beta_1$: 梯度的移动均值系数, default: 0.9, 配置名称: beta1
3. $\beta_2$: 梯度的移动均值系数, default: 0.99, 配置名称: beta2
4. $\lambda$: 权重衰减的系数, default: 0.01, 配置名称: lambda

## Install

### rocksdb安装
参考[rocksdb安装](rocksdb.md)

### damo-embedding安装

```shell
git clone https://github.com/uopensail/damo-embedding
cd damo-embedding
python setup.py install
```

### 导出格式

采用二进制的格式保存权重。具体说明如下

1. 256个int类型(4字节)：表达256个group的维度(dim)（默认值是0）

2. 256个size_t类型(8字节)：表达256个group种key的数量（默认值是0）

3. 接下来是各个key的值
   
   1. u_int64_t: 对应的key
   
   2. dim个float: 对应维度的浮点数

默认使用主机字节序，不做转换。默认情况下，x86是小端。

## Examples
参考目录../example

## Reference

[1][蚂蚁金服核心技术：百亿特征实时推荐算法揭秘](https://developer.aliyun.com/article/714366)