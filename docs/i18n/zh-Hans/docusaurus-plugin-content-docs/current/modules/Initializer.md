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

All values Are setted to 0.

#### Ones

All values Are setted to 1.

#### RandomUniform

uniform distribution, configure the following parameters:
1. min: low bound value, float type, default: -1.0
2. max: up bound value, float type, default: 1.0

#### RandomNormal

stochastic normal distribution, configure the following parameters:
1. mean: mean value, float type, default: 0.0
2. stddev: standard deviation, float type, default: 1.0

#### TruncateNormal

stochastic normal distribution, if the generated value exceeds 2 standard deviations, it is discarded and regenerate, configure the following parameters:
1. mean: mean value, float type, default: 0.0
2. stddev: standard deviation,  float type, default: 1.0

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