# Optimizer

When using an optimizer, you need to configure the `name` item to indicate which optimizer to use, and then configure their respective parameters according to different optimizer. The following are the names of each optimizer.

| Optimizer | name    |
| --------- | ------- |
| SGD       | sgd     |
| FTRL      | ftrl    |
| Adagrad   | adagrad |
| Adam      | adam    |
| AdamW     | adamw   |
| Lion      | lion    |

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
4. $\lambda$: 权重衰减的系数, default: 1e-2, 配置名称: lambda
5. $\epsilon$: 最小误差项, default: 1e-8, 配置名称: epsilon

#### Lion

[Lion](https://arxiv.org/abs/2302.06675)需要配置如下的一些参数:

1. $\eta$: 学习率, default: 3e-4, 配置名称: eta
2. $\beta_1$: 梯度的移动均值系数, default: 0.9, 配置名称: beta1
3. $\beta_2$: 梯度的移动均值系数, default: 0.99, 配置名称: beta2
4. $\lambda$: 权重衰减的系数, default: 1e-2, 配置名称: lambda

## 样例

```python
import damo
import numpy as np

# configure learning rate scheduler
schedluer_params = damo.Parameters()
schedluer_params.insert("name": "")

# configure optimizer
optimizer_params = damo.Parameters()
optimizer_params.insert("name": "sgd")
optimizer_params.insert("gamma": 0.001)
optimizer_params.insert("lambda": 0.0)

# no scheduler
opt1 = damo.PyOptimizer(optimizer_params)

# specific scheduler
opt1 = damo.PyOptimizer(optimizer_params, schedluer_params)

w = np.zeros(10, dtype=np.float32)
gs = np.random.random(10).astype(np.float32)
step = 0
opt1.call(w, gs, step)
```