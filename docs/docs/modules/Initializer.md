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

# zeros
zeros = {"name": "zeros"}

# ones
ones = {"name": "ones"}

# random_uniform
random_uniform = {"name": "random_uniform", "min": -1.0, "max": 1.0}


# random_normal
random_normal = {"name": "random_normal", "mean": 0.0, "stddev": 1.0}


# truncate_normal
truncate_normal = {"name": "truncate_normal", "mean": 0.0, "stddev": 1.0}
```