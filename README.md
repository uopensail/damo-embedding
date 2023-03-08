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
这里会将特征分成不同的group, 不同的group可以设置不同的宽度, 不同的初始化算子和优化算子。group的信息是放在key中的高8位, 即最多支持[0,256)个group。

```c++
group = Key>>56
```

#### Value
```c++
struct MetaData {
  u_int64_t key;
  u_int64_t update_time;  //更新时间
  u_int64_t update_num;   //更新次数
  int dim;
  Float data[];
};
```


#### TTL
对于一些长时间没有更新的特征, 可以通过设置TTL的方式将其删除, 这个是rocksdb自身支持的功能。这样做也是为了降低模型的大小。


### Counting Bloom Filter
Counting Bloom Filter的作用是用来过滤低频次特征。互联网的业务一般都呈现长尾的特点, 在训练机器学习/深度学习模型的时候, 长尾特征会非常多，其中大部分的长尾特征出现的频次非常低。它们对模型的收敛产生了不小的干扰: 一方面特征频次低导致这些特征无法充分训练, 另一方面它们对存储资源和计算资源消耗也非常大, 所以去掉低频次的特征就显得非常有必要。

如果是离线训练模型, 算法工程师可以进行特征的预处理, 统计特征出现的频次, 然后把这些低频次的特征去掉。但是如果是在线模型, 就无法进行特征的预处理。对稀疏特征进行处理有很多方案, 例如: 基于泊松分布的特征频次估计, 动态调整L1正则过滤等<sup>[1]</sup>。该项目提供了一种相对比较直接的方式, 利用Counting Bloom Filter来记录特征出现的次数实现特征频次过滤。需要说明的是, 项目中采用半个char来存放每一个数据, 也就是说最大的记录数就是15。因为开发者认为15这个值已经可以满足绝大部分需求了。

另外, 为了避免数据丢失的问题。开发者使用mmap的技术将内存和文件进行了映射。当这次模型训练完结或中途出现崩溃, 模型训练重启的时候, 可以通过加载文件的方式把数据恢复起来。

参考stackoverflow上的一个问题[What updates mtime after writing to memory mapped files?](https://stackoverflow.com/questions/44815329/what-updates-mtime-after-writing-to-memory-mapped-files)
> When you `mmap` a file, you're basically sharing memory directly between your process and the kernel's page cache — the same cache that holds file data that's been read from disk, or is waiting to be written to disk. A page in the page cache that's different from what's on disk (because it's been written to) is referred to as "dirty".
> There is a kernel thread that scans for dirty pages and writes them back to disk, under the control of several parameters. One important one is `dirty_expire_centisecs`. If any of the pages for a file have been dirty for longer than `dirty_expire_centisecs` then all of the dirty pages for that file will get written out. The default value is 3000 centisecs (30 seconds).

备注: 
1. 因为counting bloom filter不支持删除功能, 所以要预估好模型的容量。
2. 当capacity的值太大的时候, 会大量消耗内存和磁盘, 下面有估算的脚本。e.g: 10亿个特征,错误率在0.001的时候，消耗的内存大概在6.70G。

```python
def get_space(capacity, ffp):
    import math

    tmp = int(math.log(1.0 / ffp) * capacity / (math.log(2.0) ** 2)) >> 1
    print("%.2fG" % (tmp / (2**30)))
```
### Scheduler
不配置scheduler的时候, name为空字符串   

#### exponential_decay
配置如下的参数:    
decay_steps: float     
decay_rate: float    

#### polynomial_decay
配置如下的参数: 
end_learning_rate: float    
power: float    
decay_steps: float    


#### nature_exponential_decay
配置如下的参数:    
decay_rate: float 

#### inverse_time_decay
配置如下的参数:  
decay_steps: float  
decay_rate: float 

#### cosine_decay
配置如下的参数:   
decay_rate: float   

#### liner_cosine_decay
配置如下的参数:    
alpha: float    
beta: float    
decay_steps: float    
num_periods: float    

### Initializer

#### Zeros
0初始化

#### Ones
1初始化

#### RandomUniform
均匀分布, 需要配置如下的一些参数:    
min: 下限, default: -1.0   
max: 上限, default: 1.0

#### RandomNormal
随机正态分布, 需要配置如下的一些参数:   
mean: 均值, default: 0.0   
std: 标准差, default: 1.0

#### TruncateNormal
随机正态分布, 且2倍标准差外的数据丢弃重新生成 需要配置如下的一些参数:    
mean: 均值, default: 0.0   
std: 标准差, default: 1.0

### Optimizer

#### SGD
[SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD)需要配置如下的一些参数:
1. $\gamma$: 学习率, default: 0.001, 配置名称: gamma
2. $\lambda$: 权重衰减的系数, default: 0, 配置名称: lambda

#### FTRL
[FTRL](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/37013.pdf)需要配置如下的一些参数:
1. $\alpha$: 学习率, default: 0.005, 配置名称: alpha
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
这里展示了centos7下的安装脚本, 其他系统类似。

```shell
#!/bin/sh
yum install -y git gflags-devel snappy-devel glog-devel zlib-devel lz4-devel libzstd-devel gcc-c++ make autoreconf automake libtool cmake
cd /tmp
wget https://github.com/facebook/rocksdb/archive/v6.4.6.tar.gz
tar -xvzf v6.4.6.tar.gz
cd rocksdb-6.4.6/
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/rocksdb ..
make && make install

cat >>/etc/profile <<EOF
export CPLUS_INCLUDE_PATH=\$CPLUS_INCLUDE_PATH:/usr/local/rocksdb/include/
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/rocksdb/lib64/
export LIBRARY_PATH=\$LIBRARY_PATH:/usr/local/rocksdb/lib64/
EOF
source /etc/profile
```

### numpy添加到路径
一般来说numpy安装在了python的路径下`site-packages`文件中
```shell
NUMPY_INCLUDE_PATH=$PYTHONPATH/site-packages/numpy/core/include
NUMPY_LIBRARY_PATH=$PYTHONPATH/site-packages/numpy/core/lib

export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$NUMPY_INCLUDE_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NUMPY_LIBRARY_PATH
export LIBRARY_PATH=$LIBRARY_PATH:NUMPY_LIBRARY_PATH
```

### PyEmbedding安装
```shell
git clone https://github.com/uopensail/damo-embedding
cd damo-embedding

# to regenerate pyembedding_wrap.cxx
# swig -python -c++ pyembedding.i

python setup.py install
```

## Configuration
需要按照toml的格式进行配置, 具体示例配置如下:
```toml
[storage] # 必须配置
# 过期时间
ttl=8640000
# rocksdb数据路径
path="/tmp/embedding"
min_count=15

[filter] # 可不配置
# 容量
capacity=2147483648
# 过滤次数
count=15
# 文件存放路径
path="/tmp/filter.dat"
# 是否从文件中加载
reload=true
# 假阳性率
ffp=0.0002

[optimizer] # 必须配置
# 名字必须配置
name="sgd"
# 其他配置按照上面的参数配置
# 如果不配置则用默认参数
# 文档中没有写默认参数的键, 则必须配置否则会报错

[initializer] # 必须配置
# 名字必须配置
name="zeros"
# 其他配置按照上面的参数配置


[scheduler] # 可不配置
# 名字必须配置
name=""
# 其他配置按照上面的参数配置

```

## Examples


## Reference
[1][蚂蚁金服核心技术：百亿特征实时推荐算法揭秘](https://developer.aliyun.com/article/714366)