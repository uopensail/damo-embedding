# Initializer

When using an initializer, you need to configure the `name` item to indicate which initializer to use, and then configure their respective parameters according to different initializer. The following are the names of each initializer.

| Initializer    | name            |
|:--------------:|:---------------:|
| Zeros          | zeros           |
| Ones           | ones            |
| RandomUniform  | random_uniform  |
| RandomNormal   | random_normal   |
| TruncateNormal | truncate_normal |

#### Zeros

0初始化
需要配置如下参数:
1. name: "zeros"

#### Ones

1初始化
需要配置如下参数:
1. name: "ones"

## RandomUniform

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

### Example

```python
import damo
import numpy as np

# zero
param = damo.Parameters()
param.insert("name", "zeros")
# must be float32
value = np.random.random(10).astype(np.float32)
obj = damo.PyInitializer(param)
obj.call(value)
print("zeros: ", value)

# ones
param = damo.Parameters()
param.insert("name", "ones")
# must be float32
value = np.random.random(10).astype(np.float32)
obj = damo.PyInitializer(param)
obj.call(value)
print("ones: ", value)

# random_uniform
param = damo.Parameters()
param.insert("name", "random_uniform")
param.insert("min", -1.0)
param.insert("max", 1.0)
# must be float32
value = np.random.random(10).astype(np.float32)
obj = damo.PyInitializer(param)
obj.call(value)
print("random_uniform: ", value)

# random_normal
param = damo.Parameters()
param.insert("name", "random_normal")
param.insert("mean", 0.0)
param.insert("stddev", 1.0)
# must be float32
value = np.random.random(10).astype(np.float32)
obj = damo.PyInitializer(param)
obj.call(value)
print("random_normal: ", value)

# truncate_normal
param = damo.Parameters()
param.insert("name", "truncate_normal")
param.insert("mean", 0.0)
param.insert("stddev", 1.0)
# must be float32
value = np.random.random(10).astype(np.float32)
obj = damo.PyInitializer(param)
obj.call(value)
print("truncate_normal: ", value)
```